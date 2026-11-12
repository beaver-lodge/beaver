defmodule Beaver.MLIR.Safepoint do
  @moduledoc """
  Native safepoint liveness analysis, conflict-free frame slot coloring, resumable CFG state-machine lowering (with
  optional native LLVM frame-pointer materialization), and bounded-leaf fast-path cloning.
  """
  alias Beaver.{MLIR, Native, Walker}
  alias Beaver.MLIR.{Attribute, Block, CAPI, IRRewriter, Operation, RewriterBase, SymbolTable, Value}

  @type safepoint_entry() :: %{
          id: non_neg_integer(),
          operation: Operation.t(),
          live_values: [Value.t()],
          slots: [non_neg_integer()]
        }
  @type analysis_result() :: %{
          slot_count: non_neg_integer(),
          slot_types: [MLIR.Type.t()],
          value_slots: %{Value.t() => non_neg_integer()},
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
  Lowers func_op in place to a resumable CFG. `:safepoints` accepts op names, @callee selectors, predicates,
  operations, or mixed lists (default `["coro.suspend"]`). `:state_arg` is the resume argument index (0);
  nil emits a coro.state placeholder. `:frame_arg` optionally supplies a !llvm.ptr argument for native frame
  GEP/store/load; otherwise `:save_op`/`:load_op` select placeholders (coro.save/coro.load).
  `:slot_stride` is :auto for ABI-aligned offsets or a fixed positive byte stride.
  `:return_frame_bytes` adds the frame size to the success tuple (default false).
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

    with [region] <- func_op |> Walker.regions() |> Enum.to_list(),
         blocks when blocks != [] <- region |> Walker.blocks() |> Enum.to_list(),
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
    Native.with_resource(:digraph.new([:private]), &:digraph.delete/1, fn graph ->
      indexed = Enum.with_index(blocks)
      for {_, index} <- indexed, do: :digraph.add_vertex(graph, index)

      for {block, index} <- indexed,
          term = Block.terminator(block),
          term != nil,
          successor <- Walker.successors(term),
          target = Enum.find_index(blocks, &Block.equal?(&1, successor)),
          target != nil,
          do: :digraph.add_edge(graph, index, target)

      :digraph_utils.is_acyclic(graph)
    end)
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
        found = SymbolTable.lookup(st, clean_sym)
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
    orig_sym = Attribute.value(func_op["sym_name"])

    leaf_sym =
      Keyword.get_lazy(opts, :leaf_name, fn -> orig_sym <> "_leaf" end) |> to_string() |> String.trim_leading("@")

    case lookup_existing_leaf(func_op, leaf_sym) do
      %Operation{} = existing ->
        {existing, leaf_sym}

      nil ->
        clone = Operation.clone(func_op)
        Operation.put_attribute(clone, :sym_name, Attribute.string(leaf_sym, ctx: ctx))
        fold_fn = Keyword.get(opts, :fold_poll)

        IRRewriter.with_rewriter(ctx, fn rw ->
          insert_clone_after_if_attached(rw, func_op, clone)
          apply_poll_folding!(rw, clone, poll_names, fold_fn)
        end)

        {clone, leaf_sym}
    end
  end

  defp lookup_existing_leaf(func, sym) do
    if root = Operation.enclosing_symbol_table(func) do
      Operation.with_symbol_table(root, &lookup_leaf(&1, sym))
    end
  end

  defp lookup_leaf(st, sym) do
    op = SymbolTable.lookup(st, sym)
    unless MLIR.null?(op), do: op
  end

  defp insert_clone_after_if_attached(rw, func, clone) do
    if Operation.block(func),
      do: RewriterBase.with_insertion_point(rw, {:after, func}, fn -> RewriterBase.insert(rw, clone) end)
  end

  defp apply_poll_folding!(rw, clone, specs, callback) do
    fold = if is_function(callback, 2), do: callback, else: &fold_poll_op!(&1, clone, &2)
    Enum.each(resolve_ops(clone, specs), &fold.(rw, &1))
  end

  defp fold_poll_op!(rw, enclosing_func, poll_op) do
    results = poll_op |> Operation.results() |> Enum.to_list()

    case results do
      [] ->
        RewriterBase.erase_op(rw, poll_op)

      [res] ->
        res_type = Value.type(res)
        const_op = if Operation.name(enclosing_func) == "llvm.func", do: "llvm.mlir.constant", else: "arith.constant"

        RewriterBase.with_insertion_point(rw, {:before, poll_op}, fn ->
          zero_attr = Attribute.integer(res_type, 0)
          false_val = RewriterBase.emit(rw, poll_op, const_op, [value: zero_attr], res_type)
          RewriterBase.replace_op(rw, poll_op, false_val)
        end)

      _ ->
        raise ArgumentError,
              "cannot automatically fold multi-result poll op #{Operation.name(poll_op)}; pass :fold_poll callback"
    end
  end

  defp poll_op_cost(%Operation{} = op, cost_attr),
    do: attr_int_cost(Operation.attribute(op, cost_attr)) || operand_const_cost(op) || 1

  defp operand_const_cost(%Operation{} = op) do
    with %MLIR.Value{} = val <- Enum.at(Operation.operands(op), 0),
         {:ok, def_op} <- Value.owner(val),
         true <- Operation.name(def_op) in ["arith.constant", "llvm.mlir.constant"] do
      attr_int_cost(Operation.attribute(def_op, "value"))
    else
      _ -> nil
    end
  end

  defp attr_int_cost(nil), do: nil

  defp attr_int_cost(%MLIR.Attribute{} = attr) do
    if Attribute.integer?(attr) do
      val = Attribute.value(attr)
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
  defp normalize_slot_stride!(n) when is_integer(n) and n >= 0, do: n
  defp normalize_slot_stride!(other), do: raise(ArgumentError, "invalid :slot_stride: #{inspect(other)}")
  @doc false
  def resolve_ops(%Operation{} = _func_op, []), do: []

  def resolve_ops(%Operation{} = func_op, specs) when is_list(specs) do
    {explicit_ops, matchers} = Enum.split_with(specs, &match?(%Operation{}, &1))
    preds = Enum.map(matchers, &spec_predicate!/1)

    matched =
      if preds == [],
        do: [],
        else: Walker.find_ops(func_op, fn op -> Enum.any?(preds, & &1.(op)) end, exclude_root: true)

    explicit_ops ++ matched
  end

  def resolve_ops(%Operation{} = func_op, spec), do: resolve_ops(func_op, [spec])
  defp spec_predicate!(fun) when is_function(fun, 1), do: fn op -> if(fun.(op), do: true, else: false) end

  defp spec_predicate!("@" <> callee) when callee != "", do: fn op -> op_callee_symbol(op) == callee end

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
    attr = Operation.attribute(op, "callee")

    cond do
      is_nil(attr) -> nil
      Attribute.flat_symbol_ref?(attr) -> Attribute.value(attr)
      Attribute.string?(attr) -> attr |> Attribute.value() |> String.trim_leading("@")
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
