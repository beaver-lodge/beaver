defmodule Beaver.MLIR.Safepoint do
  @moduledoc """
  Native safepoint liveness analysis, conflict-free frame slot coloring, resumable CFG state-machine lowering (with
  optional native LLVM frame-pointer materialization), and bounded-leaf fast-path cloning.
  """

  alias Beaver.{MLIR, Native}
  alias Beaver.MLIR.{CAPI, IRRewriter, Operation, RewriterBase}

  @type safepoint_entry() :: %{
          id: non_neg_integer(),
          operation: Operation.t(),
          live_values: [MLIR.Value.t()],
          slots: [non_neg_integer()]
        }

  @type analysis_result() :: %{
          slot_count: non_neg_integer(),
          slot_types: [MLIR.Type.t()],
          value_slots: %{MLIR.Value.t() => non_neg_integer()},
          safepoints: [safepoint_entry()]
        }

  @type safepoint_matcher() ::
          Operation.t()
          | String.t()
          | atom()
          | {String.Chars.t(), String.Chars.t()}
          | (Operation.t() -> as_boolean(term()))

  @type safepoint_spec() :: safepoint_matcher() | [safepoint_matcher()]

  @doc "Computes live-across-safepoint SSA values inside `func_op` and assigns conflict-free frame slots."
  @spec analyze(Operation.t(), safepoint_spec(), keyword()) :: analysis_result()
  def analyze(%Operation{} = func_op, safepoints, opts \\ []) when is_list(opts) do
    exclude_entry_args = Keyword.get(opts, :exclude_entry_args, false)
    ops = resolve_ops(func_op, safepoints)
    CAPI.beaver_raw_safepoint_analyze(func_op.ref, Enum.map(ops, & &1.ref), exclude_entry_args) |> decode()
  end

  @doc """
  Lowers `func_op` in-place into a resumable coroutine / state-machine CFG.

  Options:
    * `:safepoints` - safepoint op name, list of op names, or `[Operation.t()]` (default `["coro.suspend"]`).
    * `:state_arg` - entry argument index holding resume state (`nil` emits `"coro.state"`, default `0`).
    * `:frame_arg` - optional entry argument index of type `!llvm.ptr`. When set, spills and reloads are
      materialized directly as `llvm.getelementptr` + `llvm.store` / `llvm.load` without unregistered ops.
    * `:slot_stride` - `:auto` (default, ABI-aligned cumulative offsets via `DataLayout`) or fixed positive
      byte stride per frame slot (e.g. `8` or `24`).
    * `:save_op` / `:load_op` - custom placeholder op names when `:frame_arg` is `nil`.
    * `:return_frame_bytes` - when `true`, returns `{:ok, func_op, frame_bytes}` instead of `{:ok, func_op}`.
  """
  @spec lower_resumable(Operation.t(), keyword()) ::
          {:ok, Operation.t()} | {:ok, Operation.t(), non_neg_integer()} | {:error, term()}
  def lower_resumable(%Operation{} = func_op, opts \\ []) when is_list(opts) do
    safepoint_spec = Keyword.get(opts, :safepoints, ["coro.suspend"])
    save_op = Keyword.get(opts, :save_op, "coro.save") |> to_string()
    load_op = Keyword.get(opts, :load_op, "coro.load") |> to_string()
    state_arg = normalize_optional_index!(Keyword.get(opts, :state_arg, 0), :state_arg)
    frame_arg = normalize_optional_index!(Keyword.get(opts, :frame_arg, nil), :frame_arg)
    slot_stride = normalize_slot_stride!(Keyword.get(opts, :slot_stride, :auto))
    ops = resolve_ops(func_op, safepoint_spec)

    try do
      {:ok, frame_bytes} =
        CAPI.beaver_raw_safepoint_lower_resumable(
          func_op.ref,
          Enum.map(ops, & &1.ref),
          save_op,
          load_op,
          {state_arg, frame_arg, slot_stride}
        )

      if Keyword.get(opts, :return_frame_bytes, false), do: {:ok, func_op, frame_bytes}, else: {:ok, func_op}
    rescue
      e -> {:error, e}
    end
  end

  @doc "Lowers `func_op` in-place into a resumable state-machine CFG, raising on failure."
  @spec lower_resumable!(Operation.t(), keyword()) :: Operation.t() | {Operation.t(), non_neg_integer()}
  def lower_resumable!(%Operation{} = func_op, opts \\ []) do
    case lower_resumable(func_op, opts) do
      {:ok, op} -> op
      {:ok, op, frame_bytes} -> {op, frame_bytes}
      {:error, reason} -> raise ArgumentError, "failed to lower resumable CFG: #{inspect(reason)}"
    end
  end

  @doc """
  Returns `true` when `func_op` has an acyclic CFG (no block back-edges and no `scf.while`/`scf.for` loops), zero
  non-poll suspend operations, and total poll cost `<= :max_work` (default `32`).
  """
  @spec bounded_leaf?(Operation.t(), keyword()) :: boolean()
  def bounded_leaf?(%Operation{} = func_op, opts \\ []) when is_list(opts) do
    poll_specs = op_specs(opts, :poll_ops, :poll, "coro.poll")
    suspend_specs = op_specs(opts, :suspend_ops, :safepoints, "coro.suspend")

    loop_ops = Keyword.get(opts, :loop_ops, ["scf.while", "scf.for"]) |> List.wrap()
    cost_attr = Keyword.get(opts, :cost_attr, "cost") |> to_string()
    max_work = Keyword.get(opts, :max_work, 32)

    with [region] <- func_op |> Beaver.Walker.regions() |> Enum.to_list(),
         blocks when blocks != [] <- region |> Beaver.Walker.blocks() |> Enum.to_list(),
         true <- acyclic_cfg?(blocks),
         [] <- resolve_ops(func_op, loop_ops),
         [] <- resolve_ops(func_op, suspend_specs) do
      total_cost =
        func_op
        |> resolve_ops(poll_specs)
        |> Enum.reduce(0, fn poll_op, acc -> acc + poll_op_cost(poll_op, cost_attr) end)

      total_cost <= max_work
    else
      _ -> false
    end
  end

  defp op_specs(opts, primary, fallback, default),
    do: opts |> Keyword.get(primary, Keyword.get(opts, fallback, [default])) |> List.wrap()

  defp acyclic_cfg?(blocks) do
    graph = :digraph.new([:private])

    try do
      indexed = Enum.with_index(blocks)
      for {_, index} <- indexed, do: :digraph.add_vertex(graph, index)

      for {block, index} <- indexed,
          term = MLIR.Block.terminator(block),
          term != nil,
          successor <- Beaver.Walker.successors(term),
          target = Enum.find_index(blocks, &MLIR.Block.equal?(&1, successor)),
          target != nil,
          do: :digraph.add_edge(graph, index, target)

      :digraph_utils.is_acyclic(graph)
    after
      :digraph.delete(graph)
    end
  end

  @doc """
  Optionally clones a bounded-leaf fast-path sibling (`<sym>_leaf` with poll safepoints folded away) and optionally
  lowers `func_op` in-place via `lower_resumable!/2` (`lower_resumable: true` by default).

  Accepts `(func_op, opts)`, `(module_or_container, func_op, opts)`, or `(module_or_container, sym_name, opts)`.
  """
  @spec bundle_leaf!(MLIR.Module.t() | Operation.t(), keyword() | Operation.t() | String.Chars.t(), keyword()) :: map()
  def bundle_leaf!(func_or_module, sym_or_opts \\ [], opts \\ [])

  def bundle_leaf!(%Operation{} = func_op, opts, []) when is_list(opts), do: do_bundle_leaf!(func_op, opts)

  def bundle_leaf!(_module_or_op, %Operation{} = func_op, opts) when is_list(opts), do: do_bundle_leaf!(func_op, opts)

  def bundle_leaf!(module_or_op, sym_name, opts) when is_list(opts) do
    root_op = Operation.from_module(module_or_op)
    clean_sym = sym_name |> to_string() |> String.trim_leading("@")

    func_op =
      Operation.with_symbol_table(root_op, fn st ->
        found = MLIR.SymbolTable.lookup(st, clean_sym)
        if MLIR.null?(found), do: raise(ArgumentError, "function @#{clean_sym} not found in symbol table")
        found
      end)

    do_bundle_leaf!(func_op, opts)
  end

  defp do_bundle_leaf!(%Operation{} = func_op, opts) do
    poll_specs = op_specs(opts, :poll_ops, :poll, "coro.poll")
    suspend_specs = op_specs(opts, :suspend_ops, :safepoints, "coro.suspend")

    eligible? = bounded_leaf?(func_op, opts)

    if Keyword.get(opts, :require_leaf, false) and not eligible? do
      raise ArgumentError, "function is not a bounded leaf"
    end

    {leaf_op, leaf_sym} = if eligible?, do: create_folded_leaf_clone!(func_op, poll_specs, opts), else: {nil, nil}

    {lowered_op, frame_bytes} =
      if Keyword.get(opts, :lower_resumable, true) do
        default_sps = Enum.uniq(poll_specs ++ suspend_specs)

        lower_opts =
          opts
          |> Keyword.drop(
            ~w(poll_ops poll suspend_ops loop_ops cost_attr max_work leaf_name fold_poll lower_resumable require_leaf)a
          )
          |> Keyword.put_new(:safepoints, default_sps)
          |> Keyword.put(:return_frame_bytes, true)

        lower_resumable!(func_op, lower_opts)
      else
        {func_op, 0}
      end

    %{
      function: lowered_op,
      leaf_function: leaf_op,
      leaf_symbol: leaf_sym,
      bounded_leaf?: eligible?,
      frame_bytes: frame_bytes
    }
  end

  defp create_folded_leaf_clone!(%Operation{} = func_op, poll_names, opts) do
    ctx = MLIR.context(func_op)
    orig_sym = func_op["sym_name"] |> MLIR.to_string() |> String.trim("\"")

    leaf_sym =
      Keyword.get_lazy(opts, :leaf_name, fn -> orig_sym <> "_leaf" end) |> to_string() |> String.trim_leading("@")

    case lookup_existing_leaf(func_op, leaf_sym) do
      %Operation{} = existing ->
        {existing, leaf_sym}

      nil ->
        clone = Operation.clone(func_op)
        Operation.put_attribute(clone, :sym_name, MLIR.Attribute.string(leaf_sym, ctx: ctx))
        fold_fn = Keyword.get(opts, :fold_poll)

        IRRewriter.with_rewriter(ctx, fn rw ->
          insert_clone_after_if_attached(rw, func_op, clone)
          apply_poll_folding!(rw, clone, poll_names, fold_fn)
        end)

        {clone, leaf_sym}
    end
  end

  defp lookup_existing_leaf(func_op, leaf_sym) do
    case Operation.enclosing_symbol_table(func_op) do
      %Operation{} = sym_op -> Operation.with_symbol_table(sym_op, &lookup_non_null_symbol(&1, leaf_sym))
      nil -> nil
    end
  end

  defp lookup_non_null_symbol(st, sym) do
    found = MLIR.SymbolTable.lookup(st, sym)
    if MLIR.null?(found), do: nil, else: found
  end

  defp insert_clone_after_if_attached(rw, func_op, clone) do
    if Operation.block(func_op) do
      RewriterBase.with_insertion_point(rw, {:after, func_op}, fn -> RewriterBase.insert(rw, clone) end)
    end
  end

  defp apply_poll_folding!(rw, clone, poll_names, fold_fn) when is_function(fold_fn, 2) do
    Enum.each(resolve_ops(clone, poll_names), &fold_fn.(rw, &1))
  end

  defp apply_poll_folding!(rw, clone, poll_names, _fold_fn) do
    Enum.each(resolve_ops(clone, poll_names), &fold_poll_op!(rw, clone, &1))
  end

  defp fold_poll_op!(rw, enclosing_func, poll_op) do
    results = poll_op |> Operation.results() |> Enum.to_list()

    case results do
      [] ->
        RewriterBase.erase_op(rw, poll_op)

      [res] ->
        res_type = MLIR.Value.type(res)
        const_op = if Operation.name(enclosing_func) == "llvm.func", do: "llvm.mlir.constant", else: "arith.constant"

        RewriterBase.with_insertion_point(rw, {:before, poll_op}, fn ->
          zero_attr = MLIR.Attribute.integer(res_type, 0)
          false_val = RewriterBase.emit(rw, poll_op, const_op, [value: zero_attr], res_type)
          RewriterBase.replace_op(rw, poll_op, false_val)
        end)

      _ ->
        raise ArgumentError,
              "cannot automatically fold multi-result poll op #{Operation.name(poll_op)}; pass :fold_poll callback"
    end
  end

  defp poll_op_cost(%Operation{} = op, cost_attr) do
    attr_int_cost(Operation.attribute(op, cost_attr)) || operand_const_cost(op) || 1
  end

  defp operand_const_cost(%Operation{} = op) do
    with %MLIR.Value{} = val <- Enum.at(Operation.operands(op), 0),
         {:ok, def_op} <- MLIR.Value.owner(val),
         true <- Operation.name(def_op) in ["arith.constant", "llvm.mlir.constant"] do
      attr_int_cost(Operation.attribute(def_op, "value"))
    else
      _ -> nil
    end
  end

  defp attr_int_cost(nil), do: nil

  defp attr_int_cost(%MLIR.Attribute{} = attr) do
    if MLIR.Attribute.integer?(attr) do
      val = MLIR.Attribute.value(attr)
      if is_integer(val) and val >= 0, do: val, else: nil
    else
      parse_non_neg_int(MLIR.to_string(attr))
    end
  end

  defp parse_non_neg_int(text) do
    case Integer.parse(text) do
      {n, _} when n >= 0 -> n
      _ -> nil
    end
  end

  defp normalize_optional_index!(nil, _key), do: -1
  defp normalize_optional_index!(idx, _key) when is_integer(idx) and idx >= 0, do: idx
  defp normalize_optional_index!(other, key), do: raise(ArgumentError, "invalid #{inspect(key)}: #{inspect(other)}")

  defp normalize_slot_stride!(:auto), do: 0
  defp normalize_slot_stride!(0), do: 0
  defp normalize_slot_stride!(n) when is_integer(n) and n > 0, do: n
  defp normalize_slot_stride!(other), do: raise(ArgumentError, "invalid :slot_stride: #{inspect(other)}")

  @doc false
  def resolve_ops(%Operation{} = _func_op, []), do: []
  def resolve_ops(%Operation{} = _func_op, [%Operation{} | _] = ops), do: ops

  def resolve_ops(%Operation{} = func_op, specs) when is_list(specs) do
    {explicit_ops, matchers} = Enum.split_with(specs, &match?(%Operation{}, &1))

    matched =
      if matchers == [] do
        []
      else
        preds = Enum.map(matchers, &spec_predicate!/1)
        Beaver.Walker.find_ops(func_op, fn op -> Enum.any?(preds, & &1.(op)) end, exclude_root: true)
      end

    explicit_ops ++ matched
  end

  def resolve_ops(%Operation{} = func_op, spec), do: resolve_ops(func_op, [spec])

  defp spec_predicate!(fun) when is_function(fun, 1), do: fn op -> if(fun.(op), do: true, else: false) end

  defp spec_predicate!("@" <> callee) when callee != "" do
    fn op -> op_callee_symbol(op) == callee end
  end

  defp spec_predicate!({op_name, callee}) do
    expected_op = to_string(op_name)
    expected_callee = callee |> to_string() |> String.trim_leading("@")
    fn op -> Operation.name(op) == expected_op and op_callee_symbol(op) == expected_callee end
  end

  defp spec_predicate!(op_name) when is_binary(op_name) or is_atom(op_name) do
    expected_op = to_string(op_name)
    fn op -> Operation.name(op) == expected_op end
  end

  defp spec_predicate!(other), do: raise(ArgumentError, "invalid safepoint operation spec: #{inspect(other)}")

  defp op_callee_symbol(%Operation{} = op) do
    case Operation.attribute(op, "callee") do
      %MLIR.Attribute{} = attr -> extract_callee_attr(attr)
      nil -> nil
    end
  end

  defp extract_callee_attr(attr) do
    cond do
      MLIR.Attribute.flat_symbol_ref?(attr) -> MLIR.Attribute.value(attr)
      MLIR.Attribute.string?(attr) -> attr |> MLIR.Attribute.value() |> String.trim_leading("@")
      true -> nil
    end
  end

  @doc false
  def decode({slot_count, raw_slot_types, raw_value_slots, raw_entries}) do
    %{
      slot_count: slot_count,
      slot_types: Enum.map(raw_slot_types, &Native.check!/1),
      value_slots: Map.new(raw_value_slots, fn {val, slot} -> {Native.check!(val), slot} end),
      safepoints:
        Enum.map(raw_entries, fn {id, op, live_values, slots} ->
          %{id: id, operation: Native.check!(op), live_values: Enum.map(live_values, &Native.check!/1), slots: slots}
        end)
    }
  end
end
