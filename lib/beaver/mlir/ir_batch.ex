defmodule Beaver.MLIR.IRBatch do
  @moduledoc """
  Bounded IR snapshots, edits, single-block builders, and multi-block CFG
  function/module builders with one native submission per batch.
  """
  alias Beaver.{Execution, MLIR, Native}

  @shorthand_types %{
    i1: "i1",
    i8: "i8",
    i16: "i16",
    i32: "i32",
    i64: "i64",
    i128: "i128",
    index: "index",
    f16: "f16",
    bf16: "bf16",
    f32: "f32",
    f64: "f64",
    ptr: "!llvm.ptr",
    none: "none"
  }

  def read(operations, opts \\ []) do
    opts = Keyword.put_new(opts, :strategy, :auto)
    names = Keyword.get(opts, :attributes, []) |> Enum.map(&to_string/1)

    case operations do
      [] ->
        []

      [first | _] ->
        ctx = MLIR.context(first)
        records = Execution.run(ctx, :snapshot, {Enum.map(operations, & &1.ref), names}, opts)

        Enum.map(records, fn {op, name, operands, results, operand_types, result_types, attributes} ->
          %{
            operation: Native.check!(op),
            name: name,
            operands: Enum.map(operands, &Native.check!/1),
            results: Enum.map(results, &Native.check!/1),
            operand_types: Enum.map(operand_types, &Native.check!/1),
            result_types: Enum.map(result_types, &Native.check!/1),
            attributes: Map.new(Enum.zip(names, Enum.map(attributes, &Native.check!/1)))
          }
        end)
    end
  end

  def update(root, edits, opts \\ []) do
    opts = Keyword.put_new(opts, :strategy, :auto)
    root = MLIR.Operation.from_module(root)

    commands =
      Enum.map(edits, fn
        {:attribute, %MLIR.Operation{ref: op}, name, %MLIR.Attribute{ref: attr}} ->
          {:attribute, op, to_string(name), attr}

        {:operand, %MLIR.Operation{ref: op}, index, %MLIR.Value{ref: value}} ->
          {:operand, op, index, value}

        edit ->
          raise ArgumentError, "unsupported IR batch edit: #{inspect(edit)}"
      end)

    {valid, diagnostics} =
      Execution.run(MLIR.context(root), :update, {root.ref, commands, Keyword.get(opts, :verifier, true)}, opts)

    if valid, do: {:ok, root, diagnostics}, else: {:error, diagnostics}
  end

  @doc "Builds and inserts a batch of up to 65,536 operations into `target` in a single NIF submission."
  def build(target, specs, opts \\ []) when is_list(specs) and is_list(opts) do
    opts = Keyword.put_new(opts, :strategy, :auto)
    {block, position} = resolve_build_target!(target, Keyword.get(opts, :position, :end))

    case specs do
      [] ->
        {:ok, [], [], []}

      _ ->
        ctx = resolve_build_context!(block, position, specs, opts)
        default_loc = Keyword.get_lazy(opts, :location, fn -> MLIR.Location.unknown(ctx: ctx) end)
        verify_root_ref = encode_verify_root(Keyword.get(opts, :verify))
        initial_bindings = Keyword.get(opts, :bindings, %{}) |> Map.new()

        {encoded_specs, _bindings} =
          specs
          |> Enum.with_index()
          |> Enum.map_reduce(initial_bindings, fn {spec, idx}, bindings ->
            encode_build_spec!(spec, idx, nil, default_loc, bindings)
          end)

        {{valid, raw_ops, raw_results}, diagnostics} =
          Execution.run(ctx, :build, {block.ref, position, encoded_specs, verify_root_ref}, opts)

        ops = Enum.map(raw_ops, &Native.check!/1)
        results = Enum.map(raw_results, fn list -> Enum.map(list, &Native.check!/1) end)
        if valid, do: {:ok, ops, results, diagnostics}, else: {:error, diagnostics}
    end
  end

  @doc "Builds a batch of operations in a single NIF submission, returning `{operations, results}` or raising."
  def build!(target, specs, opts \\ []) do
    case build(target, specs, opts) do
      {:ok, ops, results, _diagnostics} ->
        {ops, results}

      {:error, diagnostics} ->
        raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "IRBatch.build! verification failed")
    end
  end

  @doc """
  Builds a single-block or multi-block CFG function inside `target` (`MLIR.Module`, `MLIR.Operation`,
  or `MLIR.Block`), emitting all body operations across all basic blocks in a single NIF submission.
  """
  @spec build_function(
          MLIR.Module.t() | MLIR.Operation.t() | MLIR.Block.t(),
          map() | keyword() | tuple(),
          keyword()
        ) ::
          {:ok, MLIR.Operation.t(), [term()]}
          | {:ok, {MLIR.Operation.t(), map()}, [term()]}
          | {:error, [term()]}
  def build_function(target, func_spec, opts \\ [])
      when (is_map(func_spec) or is_list(func_spec) or is_tuple(func_spec)) and is_list(opts) do
    opts = Keyword.put_new(opts, :strategy, :auto)
    target_block = resolve_container_block!(target)
    ctx = resolve_container_context!(target, target_block, opts)
    spec = normalize_func_spec!(func_spec)

    loc =
      Map.get_lazy(spec, :location, fn ->
        Keyword.get_lazy(opts, :location, fn -> MLIR.Location.unknown(ctx: ctx) end)
      end)

    op_name = resolve_func_op_name!(Map.get(spec, :kind, :func))
    return_op = Map.get(spec, :return_op, Keyword.get(opts, :return_op, default_return_op(op_name)))
    sym_name = Map.fetch!(spec, :name) |> to_string() |> String.trim_leading("@")
    {input_names, input_types} = split_typed_args!(Map.get(spec, :inputs, []))
    result_types = Map.get(spec, :results, []) |> List.wrap() |> Enum.map(&normalize_type_spec!/1)

    func_op = create_function_shell!(ctx, loc, op_name, sym_name, input_types, result_types, spec)
    MLIR.Block.append(target_block, func_op)
    region = MLIR.CAPI.mlirOperationGetRegion(func_op, 0)

    block_specs = normalize_function_blocks!(spec, input_names, input_types)
    {blocks, block_index_by_name, per_block_arg_bindings} = create_cfg_blocks!(region, block_specs, ctx, loc)

    entry_arg_bindings = Map.get(per_block_arg_bindings, 0, %{})
    initial_bindings = Map.merge(entry_arg_bindings, Map.new(Map.get(spec, :bindings, [])))

    {encoded_specs, final_bindings} =
      encode_cfg_blocks!(
        block_specs,
        blocks,
        block_index_by_name,
        per_block_arg_bindings,
        return_op,
        loc,
        initial_bindings
      )

    verify_target =
      case Keyword.get(opts, :verify, false) do
        true -> func_op
        other -> other
      end

    verify_root_ref = encode_verify_root(verify_target)

    case encoded_specs do
      [] ->
        state = %{operations: [], results: [], blocks: blocks, by_name: block_index_by_name, bindings: final_bindings}
        finalize_function_result(func_op, state, [], true, opts)

      _ ->
        block_refs = Enum.map(blocks, & &1.ref)

        {{valid, raw_ops, raw_results}, diagnostics} =
          Execution.run(ctx, :build, {block_refs, :blocks, encoded_specs, verify_root_ref}, opts)

        ops = Enum.map(raw_ops, &Native.check!/1)
        results = Enum.map(raw_results, fn list -> Enum.map(list, &Native.check!/1) end)

        state = %{
          operations: ops,
          results: results,
          blocks: blocks,
          by_name: block_index_by_name,
          bindings: final_bindings
        }

        finalize_function_result(func_op, state, diagnostics, valid, opts)
    end
  end

  @doc "Builds a function via `build_function/3`, returning the created `MLIR.Operation` (or `{op, details}`) or raising."
  def build_function!(target, func_spec, opts \\ []) do
    case build_function(target, func_spec, opts) do
      {:ok, result, _diagnostics} ->
        result

      {:error, diagnostics} ->
        raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "IRBatch.build_function! verification failed")
    end
  end

  @doc """
  Creates a new `MLIR.Module` in `ctx` and populates it with `func_specs` using `build_function/3`.
  Verifies the resulting module once at the end by default (`verify: true`).
  """
  @spec build_module(MLIR.Context.t(), [map() | keyword()], keyword()) ::
          {:ok, MLIR.Module.t(), [term()]} | {:error, [term()]}
  def build_module(%MLIR.Context{} = ctx, func_specs, opts \\ []) when is_list(func_specs) and is_list(opts) do
    loc = Keyword.get_lazy(opts, :location, fn -> MLIR.Location.unknown(ctx: ctx) end)
    module = MLIR.Module.empty(ctx: ctx, location: loc)
    per_fn_opts = opts |> Keyword.put(:ctx, ctx) |> Keyword.put(:verify, false) |> Keyword.delete(:return_details)

    try do
      Enum.each(func_specs, fn func_spec ->
        {:ok, _fn_op, _} = build_function(module, func_spec, per_fn_opts)
      end)

      if Keyword.get(opts, :verify, true) do
        case MLIR.verify(module) do
          {:ok, _} ->
            {:ok, module, []}

          {:error, diagnostics} ->
            MLIR.Module.destroy(module)
            {:error, diagnostics}
        end
      else
        {:ok, module, []}
      end
    rescue
      e ->
        MLIR.Module.destroy(module)
        reraise e, __STACKTRACE__
    end
  end

  @doc "Builds and verifies an `MLIR.Module` from `func_specs`, raising on failure."
  @spec build_module!(MLIR.Context.t(), [map() | keyword()], keyword()) :: MLIR.Module.t()
  def build_module!(%MLIR.Context{} = ctx, func_specs, opts \\ []) do
    case build_module(ctx, func_specs, opts) do
      {:ok, module, _diagnostics} ->
        module

      {:error, diagnostics} ->
        raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "IRBatch.build_module! verification failed")
    end
  end

  defp finalize_function_result(func_op, state, diagnostics, true, opts) do
    if Keyword.get(opts, :return_details, false) do
      block_map = Map.new(state.by_name, fn {k, idx} -> {k, Enum.at(state.blocks, idx)} end)
      details = %{operations: state.operations, results: state.results, blocks: block_map, bindings: state.bindings}
      {:ok, {func_op, details}, diagnostics}
    else
      {:ok, func_op, diagnostics}
    end
  end

  defp finalize_function_result(_func_op, _state, diagnostics, false, _opts),
    do: {:error, diagnostics}

  defp normalize_func_spec!(%{} = map), do: map
  defp normalize_func_spec!(kw) when is_list(kw), do: Map.new(kw)

  defp normalize_func_spec!({kind, name, inputs, results}),
    do: %{kind: kind, name: name, inputs: inputs, results: results}

  defp normalize_func_spec!({kind, name, inputs, results, extra}) when is_list(extra) or is_map(extra) do
    extra
    |> Map.new()
    |> Map.merge(%{kind: kind, name: name, inputs: inputs, results: results})
  end

  defp normalize_func_spec!(other),
    do: raise(ArgumentError, "invalid function specification for IRBatch.build_function: #{inspect(other)}")

  defp resolve_container_block!(%MLIR.Block{} = block), do: block
  defp resolve_container_block!(%MLIR.Module{} = module), do: MLIR.Module.body(module)

  defp resolve_container_block!(%MLIR.Operation{} = op) do
    MLIR.Module.body(op) || raise(ArgumentError, "target operation does not have an entry block")
  end

  defp resolve_container_block!(other),
    do: raise(ArgumentError, "invalid build_function target: #{inspect(other)}")

  defp resolve_container_context!(%MLIR.Module{} = mod, _blk, _opts), do: MLIR.context(mod)
  defp resolve_container_context!(%MLIR.Operation{} = op, _blk, _opts), do: MLIR.context(op)

  defp resolve_container_context!(_target, %MLIR.Block{} = blk, opts) do
    opts[:ctx] ||
      case MLIR.Block.parent_operation(blk) do
        %MLIR.Operation{} = op -> MLIR.context(op)
        nil -> raise ArgumentError, "unable to infer MLIR.Context for detached block; pass :ctx in opts"
      end
  end

  defp resolve_func_op_name!(:func), do: "func.func"
  defp resolve_func_op_name!(:llvm), do: "llvm.func"
  defp resolve_func_op_name!(name) when is_binary(name) and byte_size(name) > 0, do: name
  defp resolve_func_op_name!(other), do: raise(ArgumentError, "invalid function :kind #{inspect(other)}")

  defp default_return_op("llvm.func"), do: "llvm.return"
  defp default_return_op(_), do: "func.return"

  defp split_typed_args!(args) when is_list(args) do
    Enum.map(args, fn
      {name, type} when is_atom(name) or is_binary(name) -> {name, normalize_type_spec!(type)}
      type -> {nil, normalize_type_spec!(type)}
    end)
    |> Enum.unzip()
  end

  defp create_function_shell!(ctx, loc, op_name, sym_name, input_types, result_types, spec) do
    base_attrs = build_func_base_attrs!(ctx, op_name, sym_name, input_types, result_types, spec)
    extra_attrs = Enum.map(Map.get(spec, :attributes, []), fn {k, v} -> {normalize_attr_key(k), v} end)

    func_op =
      %Beaver.Changeset{name: op_name, location: loc, context: ctx}
      |> Beaver.Changeset.add_argument(base_attrs ++ extra_attrs)
      |> Beaver.Changeset.add_argument(MLIR.Region.create())
      |> MLIR.Operation.create()

    maybe_set_non_func_visibility(func_op, op_name, Map.get(spec, :visibility))
    func_op
  end

  defp normalize_attr_key(k) when is_atom(k), do: k
  defp normalize_attr_key(k), do: String.to_atom(to_string(k))

  defp build_func_base_attrs!(ctx, op_name, sym_name, input_types, result_types, spec) do
    include_ft = Map.get(spec, :include_function_type, op_name in ["func.func", "llvm.func"])

    [sym_name: MLIR.Attribute.string(sym_name, ctx: ctx)]
    |> maybe_put_function_type(include_ft, ctx, op_name, input_types, result_types, Map.get(spec, :function_type))
    |> maybe_put_func_visibility!(ctx, op_name, Map.get(spec, :visibility))
  end

  defp maybe_put_function_type(attrs, false, _ctx, _op_name, _in_types, _out_types, _explicit), do: attrs

  defp maybe_put_function_type(attrs, true, ctx, op_name, input_types, result_types, explicit) do
    ft = build_function_type!(ctx, op_name, input_types, result_types, explicit)
    Keyword.put(attrs, :function_type, MLIR.Attribute.type(ft))
  end

  defp maybe_put_func_visibility!(attrs, _ctx, _op_name, nil), do: attrs
  defp maybe_put_func_visibility!(attrs, _ctx, _op_name, :public), do: attrs

  defp maybe_put_func_visibility!(attrs, ctx, "func.func", vis) when vis in [:private, :nested],
    do: Keyword.put(attrs, :sym_visibility, MLIR.Attribute.string(to_string(vis), ctx: ctx))

  defp maybe_put_func_visibility!(attrs, _ctx, _op_name, vis) when vis in [:private, :nested], do: attrs

  defp maybe_put_func_visibility!(_attrs, _ctx, _op_name, other),
    do: raise(ArgumentError, "invalid function :visibility #{inspect(other)}")

  defp maybe_set_non_func_visibility(_func_op, _op_name, vis) when vis in [nil, :public], do: :ok
  defp maybe_set_non_func_visibility(_func_op, "func.func", _vis), do: :ok
  defp maybe_set_non_func_visibility(func_op, _op_name, vis), do: MLIR.SymbolTable.set_visibility(func_op, vis)

  defp build_function_type!(ctx, _op_name, _inputs, _results, explicit) when is_binary(explicit),
    do: MLIR.Type.get(explicit, ctx: ctx)

  defp build_function_type!(_ctx, _op_name, _inputs, _results, %MLIR.Type{} = explicit), do: explicit

  defp build_function_type!(ctx, "llvm.func", input_types, result_types, _) do
    in_str = Enum.map_join(input_types, ", ", &type_spec_to_string(&1, ctx))

    ret_str =
      case result_types do
        [] -> "!llvm.void"
        [single] -> type_spec_to_string(single, ctx)
        many -> "!llvm.struct<(#{Enum.map_join(many, ", ", &type_spec_to_string(&1, ctx))})>"
      end

    MLIR.Type.get("!llvm.func<#{ret_str} (#{in_str})>", ctx: ctx)
  end

  defp build_function_type!(ctx, _op_name, input_types, result_types, _) do
    ins = Enum.map(input_types, &resolve_mlir_type(&1, ctx))
    outs = Enum.map(result_types, &resolve_mlir_type(&1, ctx))
    MLIR.Type.function(ins, outs, ctx: ctx)
  end

  defp resolve_mlir_type(%MLIR.Type{} = t, _ctx), do: t
  defp resolve_mlir_type(s, ctx) when is_binary(s), do: MLIR.Type.get(s, ctx: ctx)

  defp type_spec_to_string(%MLIR.Type{} = t, _ctx), do: MLIR.to_string(t)
  defp type_spec_to_string(s, _ctx) when is_binary(s), do: s

  defp normalize_function_blocks!(spec, input_names, input_types) do
    cond do
      Map.has_key?(spec, :blocks) and Map.has_key?(spec, :body) ->
        raise ArgumentError, "function spec accepts either :body or :blocks, not both"

      Map.has_key?(spec, :blocks) ->
        blocks = Map.fetch!(spec, :blocks)
        if blocks == [], do: raise(ArgumentError, ":blocks cannot be empty")

        blocks
        |> Enum.with_index()
        |> Enum.map(fn {blk, idx} ->
          normalize_block_entry!(blk, idx, input_names, input_types)
        end)

      true ->
        [%{name: :entry, arg_names: input_names, arg_types: input_types, body: Map.get(spec, :body, [])}]
    end
  end

  defp normalize_block_entry!({name, {args, body}}, idx, input_names, input_types)
       when (is_atom(name) or is_binary(name)) and is_list(args) and is_list(body) do
    {arg_names, arg_types} =
      if idx == 0 and args == [] and input_types != [],
        do: {input_names, input_types},
        else: split_typed_args!(args)

    %{name: name, arg_names: arg_names, arg_types: arg_types, body: body}
  end

  defp normalize_block_entry!({name, body}, idx, input_names, input_types)
       when (is_atom(name) or is_binary(name)) and is_list(body) do
    {arg_names, arg_types} = if idx == 0, do: {input_names, input_types}, else: {[], []}
    %{name: name, arg_names: arg_names, arg_types: arg_types, body: body}
  end

  defp normalize_block_entry!(blk, idx, input_names, input_types) when is_map(blk) or is_list(blk) do
    bmap = Map.new(blk)
    name = Map.get(bmap, :name, :"bb#{idx}")

    {arg_names, arg_types} =
      if idx == 0 and not Map.has_key?(bmap, :args),
        do: {input_names, input_types},
        else: split_typed_args!(Map.get(bmap, :args, []))

    %{name: name, arg_names: arg_names, arg_types: arg_types, body: Map.get(bmap, :body, [])}
  end

  defp create_cfg_blocks!(region, block_specs, ctx, loc) do
    {blocks, by_name, per_block_arg_bindings} =
      block_specs
      |> Enum.with_index()
      |> Enum.reduce({[], %{}, %{}}, fn {%{name: bname, arg_names: anames, arg_types: atypes}, b_idx},
                                        {blks, bmap, per_blk_binds} ->
        if Map.has_key?(bmap, bname),
          do: raise(ArgumentError, "duplicate block name in IRBatch.build_function: #{inspect(bname)}")

        resolved_types = Enum.map(atypes, &resolve_mlir_type(&1, ctx))
        blk = MLIR.Region.append_block(region, resolved_types, ctx: ctx, location: loc)

        blk_binds =
          anames
          |> Enum.with_index()
          |> Enum.reduce(%{}, fn
            {nil, _a_idx}, acc -> acc
            {:_, _a_idx}, acc -> acc
            {aname, a_idx}, acc -> Map.put(acc, aname, {:cfg_block_arg, b_idx, a_idx})
          end)

        {[blk | blks], Map.put(bmap, bname, b_idx), Map.put(per_blk_binds, b_idx, blk_binds)}
      end)

    {Enum.reverse(blocks), by_name, per_block_arg_bindings}
  end

  defp encode_cfg_blocks!(
         block_specs,
         blocks,
         block_index_by_name,
         per_block_arg_bindings,
         return_op,
         default_loc,
         initial_bindings
       ) do
    {rev_specs, {_next_idx, final_bindings}} =
      block_specs
      |> Enum.with_index()
      |> Enum.reduce({[], {0, initial_bindings}}, fn {%{body: body}, b_idx}, {acc_specs, {op_idx, binds}} ->
        block_binds = Map.merge(binds, Map.get(per_block_arg_bindings, b_idx, %{}))

        Enum.reduce(body, {acc_specs, {op_idx, block_binds}}, fn raw_stmt, {inner_acc, {cur_idx, cur_binds}} ->
          expanded = expand_cfg_statement!(raw_stmt, blocks, block_index_by_name, return_op)
          {encoded, next_binds} = encode_build_spec!(expanded, cur_idx, b_idx, default_loc, cur_binds)
          {[encoded | inner_acc], {cur_idx + 1, next_binds}}
        end)
      end)

    {Enum.reverse(rev_specs), final_bindings}
  end

  defp expand_cfg_statement!(:return, _blocks, _by_name, return_op),
    do: %{name: return_op, operands: []}

  defp expand_cfg_statement!({:return, operands}, _blocks, _by_name, return_op),
    do: %{name: return_op, operands: List.wrap(operands)}

  defp expand_cfg_statement!({:br, dest_spec}, blocks, by_name, _return_op) do
    {blk, args} = normalize_branch_dest!(dest_spec, blocks, by_name)
    %{name: "cf.br", operands: args, successors: [blk]}
  end

  defp expand_cfg_statement!({:br, target, args}, blocks, by_name, _return_op) when is_list(args) do
    %{name: "cf.br", operands: args, successors: [lookup_cfg_block!(target, blocks, by_name)]}
  end

  defp expand_cfg_statement!({:cond_br, cond_ref, true_dest, false_dest}, blocks, by_name, _return_op) do
    {t_blk, t_args} = normalize_branch_dest!(true_dest, blocks, by_name)
    {f_blk, f_args} = normalize_branch_dest!(false_dest, blocks, by_name)

    %{
      name: "cf.cond_br",
      operands: [cond_ref] ++ t_args ++ f_args,
      successors: [t_blk, f_blk],
      attributes: [operandSegmentSizes: {:dense_i32, [1, length(t_args), length(f_args)]}]
    }
  end

  defp expand_cfg_statement!({:switch, {flag_type, flag_ref}, default_dest, cases}, blocks, by_name, return_op)
       when is_list(cases),
       do: expand_cfg_statement!({:switch, flag_ref, flag_type, default_dest, cases}, blocks, by_name, return_op)

  defp expand_cfg_statement!({:switch, flag_ref, flag_type, default_dest, cases}, blocks, by_name, _return_op)
       when is_list(cases) do
    {def_blk, def_args} = normalize_branch_dest!(default_dest, blocks, by_name)
    flag_type_str = normalize_type_string!(flag_type)

    norm_cases =
      Enum.map(cases, fn
        {val, dest_spec} when is_integer(val) ->
          {blk, args} = normalize_branch_dest!(dest_spec, blocks, by_name)
          {val, blk, args}

        {val, dest_name, args} when is_integer(val) and is_list(args) ->
          {val, lookup_cfg_block!(dest_name, blocks, by_name), args}

        other ->
          raise ArgumentError, "invalid :switch case specification: #{inspect(other)}"
      end)

    case_args_flat = Enum.flat_map(norm_cases, &elem(&1, 2))
    case_seg_sizes = Enum.map(norm_cases, fn {_, _, args} -> length(args) end)
    case_blocks = Enum.map(norm_cases, &elem(&1, 1))

    attrs = [
      operandSegmentSizes: {:dense_i32, [1, length(def_args), length(case_args_flat)]},
      case_operand_segments: {:dense_i32, case_seg_sizes}
    ]

    attrs =
      if norm_cases == [] do
        attrs
      else
        vals_str = Enum.map_join(norm_cases, ", ", &Integer.to_string(elem(&1, 0)))

        Keyword.put(
          attrs,
          :case_values,
          {:parse, "dense<[#{vals_str}]> : vector<#{length(norm_cases)}x#{flag_type_str}>"}
        )
      end

    %{
      name: "cf.switch",
      operands: [flag_ref] ++ def_args ++ case_args_flat,
      successors: [def_blk | case_blocks],
      attributes: attrs
    }
  end

  defp expand_cfg_statement!(spec, blocks, by_name, _return_op) when is_map(spec) or is_list(spec) do
    map = Map.new(spec)

    if Map.has_key?(map, :successors) do
      Map.update!(map, :successors, fn succs -> Enum.map(succs, &lookup_cfg_block!(&1, blocks, by_name)) end)
    else
      map
    end
  end

  defp expand_cfg_statement!(other, _blocks, _by_name, _return_op),
    do: raise(ArgumentError, "unsupported CFG statement in IRBatch.build_function: #{inspect(other)}")

  defp normalize_branch_dest!({dest, args}, blocks, by_name) when is_list(args),
    do: {lookup_cfg_block!(dest, blocks, by_name), args}

  defp normalize_branch_dest!(dest, blocks, by_name),
    do: {lookup_cfg_block!(dest, blocks, by_name), []}

  defp lookup_cfg_block!(%MLIR.Block{} = blk, _blocks, _by_name), do: blk

  defp lookup_cfg_block!(name, blocks, by_name) do
    case Map.fetch(by_name, name) do
      {:ok, idx} -> Enum.at(blocks, idx)
      :error -> raise ArgumentError, "unknown CFG block #{inspect(name)} (known: #{inspect(Map.keys(by_name))})"
    end
  end

  defp encode_verify_root(nil), do: nil
  defp encode_verify_root(false), do: nil
  defp encode_verify_root(root), do: MLIR.Operation.from_module(root).ref

  defp resolve_build_target!(%MLIR.Block{} = block, pos) when pos in [:end, :start], do: {block, pos}

  defp resolve_build_target!(%MLIR.Block{} = block, {rel, %MLIR.Operation{ref: op_ref}})
       when rel in [:before, :after],
       do: {block, {rel, op_ref}}

  defp resolve_build_target!({rel, %MLIR.Operation{} = op}, _pos) when rel in [:before, :after] do
    case MLIR.Operation.block(op) do
      nil -> raise ArgumentError, "anchor operation does not belong to a block"
      block -> {block, {rel, op.ref}}
    end
  end

  defp resolve_build_target!(other, pos),
    do: raise(ArgumentError, "invalid IRBatch.build target #{inspect(other)} with position #{inspect(pos)}")

  defp resolve_build_context!(block, position, specs, opts) do
    cond do
      opts[:ctx] ->
        opts[:ctx]

      opts[:location] ->
        MLIR.context(opts[:location])

      match?({_rel, _op_ref}, position) ->
        {_rel, op_ref} = position
        MLIR.context(%MLIR.Operation{ref: op_ref})

      true ->
        case MLIR.Block.parent_operation(block) do
          nil -> find_spec_context!(specs)
          parent_op -> MLIR.context(parent_op)
        end
    end
  end

  defp find_spec_context!(specs) do
    Enum.find_value(specs, &spec_context(Map.new(&1))) ||
      raise(ArgumentError, "unable to infer MLIR.Context for IRBatch.build; pass :ctx in opts")
  end

  defp spec_context(%{location: %MLIR.Location{} = loc}), do: MLIR.context(loc)

  defp spec_context(map) do
    attrs = Map.get(map, :attributes, []) |> Enum.map(fn {_, v} -> v end)
    items = Map.get(map, :results, Map.get(map, :result_types, [])) ++ Map.get(map, :operands, []) ++ attrs

    Enum.find_value(items, fn
      %mod{} = h when mod in [MLIR.Type, MLIR.Value, MLIR.Attribute] -> MLIR.context(h)
      {:type, %MLIR.Type{} = t} -> MLIR.context(t)
      _ -> nil
    end)
  end

  defp encode_build_spec!(spec, op_idx, block_idx, default_loc, bindings) when is_map(spec) or is_list(spec) do
    map = Map.new(spec)
    name = Map.fetch!(map, :name) |> to_string()
    %MLIR.Location{ref: loc_ref} = Map.get(map, :location, default_loc)

    operands = Enum.map(Map.get(map, :operands, []), &encode_operand_ref!(&1, bindings))

    raw_results = Map.get(map, :results, Map.get(map, :result_types, []))
    result_types = Enum.map(List.wrap(raw_results), &encode_type_term!/1)

    attributes = Enum.map(Map.get(map, :attributes, []), &encode_attribute_pair!/1)

    successors =
      Enum.map(Map.get(map, :successors, []), fn
        %MLIR.Block{ref: ref} -> ref
        other -> raise ArgumentError, "unsupported IRBatch.build successor: #{inspect(other)}"
      end)

    regions = Map.get(map, :regions, 0)
    unless is_integer(regions) and regions >= 0, do: raise(ArgumentError, ":regions must be a non-negative integer")

    next_bindings = register_spec_bindings!(bindings, Map.get(map, :bind), op_idx, length(result_types))

    encoded =
      if is_nil(block_idx),
        do: {name, loc_ref, operands, result_types, attributes, successors, regions},
        else: {name, loc_ref, operands, result_types, attributes, successors, regions, block_idx}

    {encoded, next_bindings}
  end

  defp encode_operand_ref!(%MLIR.Value{ref: ref}, _bindings), do: {0, 0, ref}

  defp encode_operand_ref!({:op, op_idx}, _bindings) when is_integer(op_idx) and op_idx >= 0,
    do: {1, op_idx, 0}

  defp encode_operand_ref!({:result, op_idx, res_idx}, _bindings)
       when is_integer(op_idx) and op_idx >= 0 and is_integer(res_idx) and res_idx >= 0,
       do: {1, op_idx, res_idx}

  defp encode_operand_ref!({:block_arg, arg_idx}, _bindings) when is_integer(arg_idx) and arg_idx >= 0,
    do: {2, arg_idx, 0}

  defp encode_operand_ref!({:block_arg, blk_idx, arg_idx}, _bindings)
       when is_integer(blk_idx) and blk_idx >= 0 and is_integer(arg_idx) and arg_idx >= 0,
       do: {3, blk_idx, arg_idx}

  defp encode_operand_ref!(sym, bindings) when is_atom(sym) or is_binary(sym) do
    case Map.fetch(bindings, sym) do
      {:ok, {:cfg_block_arg, blk_idx, arg_idx}} -> {3, blk_idx, arg_idx}
      {:ok, {:op_result, op_idx, res_idx}} -> {1, op_idx, res_idx}
      {:ok, %MLIR.Value{ref: ref}} -> {0, 0, ref}
      :error -> raise ArgumentError, "unbound SSA reference in IRBatch: #{inspect(sym)}"
    end
  end

  defp encode_operand_ref!(other, _bindings),
    do: raise(ArgumentError, "unsupported IRBatch.build operand reference: #{inspect(other)}")

  defp register_spec_bindings!(bindings, nil, _op_idx, _num_results), do: bindings

  defp register_spec_bindings!(bindings, name, op_idx, num_results) when is_atom(name) or is_binary(name) do
    if num_results < 1,
      do: raise(ArgumentError, "cannot bind #{inspect(name)} on 0-result operation at index #{op_idx}")

    if name == :_, do: bindings, else: Map.put(bindings, name, {:op_result, op_idx, 0})
  end

  defp register_spec_bindings!(bindings, names, op_idx, num_results) when is_list(names) do
    if length(names) != num_results do
      raise ArgumentError,
            "expected #{num_results} binding names for operation at index #{op_idx}, got #{length(names)}"
    end

    names
    |> Enum.with_index()
    |> Enum.reduce(bindings, fn
      {nil, _}, acc -> acc
      {:_, _}, acc -> acc
      {name, r_idx}, acc when is_atom(name) or is_binary(name) -> Map.put(acc, name, {:op_result, op_idx, r_idx})
      {other, _}, _acc -> raise ArgumentError, "invalid binding name: #{inspect(other)}"
    end)
  end

  defp normalize_type_spec!(%MLIR.Type{} = t), do: t
  defp normalize_type_spec!(a) when is_map_key(@shorthand_types, a), do: Map.fetch!(@shorthand_types, a)
  defp normalize_type_spec!(s) when is_binary(s) and byte_size(s) > 0, do: s
  defp normalize_type_spec!(other), do: raise(ArgumentError, "unsupported type specification: #{inspect(other)}")

  defp normalize_type_string!(a) when is_map_key(@shorthand_types, a), do: Map.fetch!(@shorthand_types, a)
  defp normalize_type_string!(s) when is_binary(s) and byte_size(s) > 0, do: s
  defp normalize_type_string!(%MLIR.Type{} = t), do: MLIR.to_string(t)

  defp normalize_type_string!(other),
    do: raise(ArgumentError, "unsupported integer type specification: #{inspect(other)}")

  defp encode_type_term!(%MLIR.Type{ref: ref}), do: ref
  defp encode_type_term!(a) when is_map_key(@shorthand_types, a), do: Map.fetch!(@shorthand_types, a)
  defp encode_type_term!(s) when is_binary(s) and byte_size(s) > 0, do: s
  defp encode_type_term!(other), do: raise(ArgumentError, "unsupported IRBatch.build result type: #{inspect(other)}")

  defp encode_attribute_pair!({attr_name, value}) do
    {to_string(attr_name), encode_attribute_value!(value)}
  end

  defp encode_attribute_pair!(other),
    do: raise(ArgumentError, "unsupported IRBatch.build attribute: #{inspect(other)}")

  defp encode_attribute_value!(%MLIR.Attribute{ref: ref}), do: ref
  defp encode_attribute_value!(:unit), do: :unit
  defp encode_attribute_value!(b) when is_boolean(b), do: b
  defp encode_attribute_value!(s) when is_binary(s) and byte_size(s) > 0, do: s
  defp encode_attribute_value!({:unit}), do: :unit
  defp encode_attribute_value!({tag, n}) when tag in [:i8, :i16, :i32, :i64, :index] and is_integer(n), do: {tag, n}
  defp encode_attribute_value!({:int, w, n}) when is_integer(w) and w > 0 and is_integer(n), do: {:int, w, n}

  defp encode_attribute_value!({tag, v}) when tag in [:bool, :i1] and (is_boolean(v) or v in [0, 1]),
    do: {tag, v}

  defp encode_attribute_value!({tag, x}) when tag in [:f32, :f64] and is_number(x), do: {tag, x * 1.0}
  defp encode_attribute_value!({:string, s}) when is_binary(s), do: {:string, s}

  defp encode_attribute_value!({tag, s}) when tag in [:symbol, :flat_symbol] and (is_binary(s) or is_atom(s)),
    do: {:symbol, s |> to_string() |> String.trim_leading("@")}

  defp encode_attribute_value!({:type, t}), do: {:type, encode_type_term!(t)}

  defp encode_attribute_value!({tag, list}) when tag in [:dense_i32, :dense_i64] and is_list(list) do
    unless Enum.all?(list, &is_integer/1),
      do: raise(ArgumentError, "#{inspect(tag)} requires a list of integers, got: #{inspect(list)}")

    {tag, list}
  end

  defp encode_attribute_value!({:parse, s}) when is_binary(s) and byte_size(s) > 0, do: {:parse, s}

  defp encode_attribute_value!(other),
    do: raise(ArgumentError, "unsupported IRBatch.build attribute value: #{inspect(other)}")
end
