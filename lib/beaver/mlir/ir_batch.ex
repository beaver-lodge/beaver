defmodule Beaver.MLIR.IRBatch do
  @moduledoc """
  Bounded IR snapshots, edits, single-block builders, and multi-block CFG function/module builders with one native
  submission per batch.
  """
  alias Beaver.{Execution, MLIR, Native}
  @shorthand_types Beaver.MLIR.IRBatch.Spec.types()
  def read(operations, opts \\ [])
  def read([], _opts), do: []

  def read([first | _] = operations, opts) do
    names = Enum.map(Keyword.get(opts, :attributes, []), &to_string/1)
    payload = {Enum.map(operations, & &1.ref), names}
    records = Execution.run(MLIR.context(first), :snapshot, payload, Keyword.put_new(opts, :strategy, :auto))

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

  @doc "Builds and inserts a batch of up to 1,048,576 operations into `target` in a single NIF submission."
  def build(target, specs, opts \\ []) when is_list(specs) and is_list(opts) do
    opts = Keyword.put_new(opts, :strategy, :auto)
    {block, position} = resolve_build_target!(target, Keyword.get(opts, :position, :end))

    if specs == [] do
      {:ok, [], [], []}
    else
      ctx = resolve_build_context!(block, position, specs, opts)
      loc = Keyword.get_lazy(opts, :location, fn -> MLIR.Location.unknown(ctx: ctx) end)
      bindings = Map.new(Keyword.get(opts, :bindings, %{}))

      {encoded, _} =
        Enum.map_reduce(Enum.with_index(specs), bindings, fn {spec, i}, acc ->
          encode_build_spec!(spec, i, nil, loc, acc)
        end)

      submit_build(ctx, block.ref, position, encoded, encode_verify_root(opts[:verify]), opts)
    end
  end

  @doc """
  Builds a single-block or multi-block CFG function inside `target` (`MLIR.Module`, `MLIR.Operation`, or
  `MLIR.Block`), emitting all body operations across all basic blocks in a single NIF submission.
  """
  @spec build_function(MLIR.Module.t() | MLIR.Operation.t() | MLIR.Block.t(), map() | keyword() | tuple(), keyword()) ::
          {:ok, MLIR.Operation.t(), [term()]} | {:ok, {MLIR.Operation.t(), map()}, [term()]} | {:error, [term()]}
  def build_function(target, func_spec, opts \\ [])
      when (is_map(func_spec) or is_list(func_spec) or is_tuple(func_spec)) and is_list(opts) do
    opts = Keyword.put_new(opts, :strategy, :auto)
    target_block = resolve_container_block!(target)
    ctx = resolve_container_context!(target, target_block, opts)
    spec = __MODULE__.Spec.function(func_spec)

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
    plan = create_cfg_blocks!(region, block_specs, ctx, loc)
    blocks = Map.new(plan, &{&1.name, &1.block})
    initial = Map.new(Map.get(spec, :bindings, []))
    {encoded_specs, final_bindings} = encode_cfg_blocks!(plan, blocks, return_op, loc, initial)
    verify = if opts[:verify] == true, do: func_op, else: opts[:verify]
    refs = Enum.map(plan, & &1.block.ref)

    with {:ok, ops, results, diagnostics} <-
           submit_build(ctx, refs, :blocks, encoded_specs, encode_verify_root(verify), opts) do
      details = %{operations: ops, results: results, blocks: blocks, bindings: final_bindings}
      {:ok, if(opts[:return_details], do: {func_op, details}, else: func_op), diagnostics}
    end
  end

  defp submit_build(_ctx, _blocks, _position, [], _verify, _opts), do: {:ok, [], [], []}

  defp submit_build(ctx, blocks, position, specs, verify, opts) do
    {{valid, ops, results}, diagnostics} = Execution.run(ctx, :build, {blocks, position, specs, verify}, opts)

    if valid,
      do:
        {:ok, Enum.map(ops, &Native.check!/1), Enum.map(results, &Enum.map(&1, fn v -> Native.check!(v) end)),
         diagnostics},
      else: {:error, diagnostics}
  end

  @doc """
  Creates a new `MLIR.Module` in `ctx` and populates it with `func_specs` using `build_function/3`. Verifies the
  resulting module once at the end by default (`verify: true`). An optional `:finalize` callback can insert
  declarations or imports before verification; failure destroys the partially built module.
  """
  @spec build_module(MLIR.Context.t(), [map() | keyword()], keyword()) ::
          {:ok, MLIR.Module.t(), [term()]} | {:error, [term()]}
  def build_module(%MLIR.Context{} = ctx, func_specs, opts \\ []) when is_list(func_specs) and is_list(opts) do
    loc = Keyword.get_lazy(opts, :location, fn -> MLIR.Location.unknown(ctx: ctx) end)
    module = MLIR.Module.empty(ctx: ctx, location: loc)
    per_fn_opts = opts |> Keyword.put(:ctx, ctx) |> Keyword.put(:verify, false) |> Keyword.delete(:return_details)

    try do
      Enum.each(func_specs, fn func_spec -> {:ok, _fn_op, _} = build_function(module, func_spec, per_fn_opts) end)
      if finalize = opts[:finalize], do: finalize.(module)
      verified = if Keyword.get(opts, :verify, true), do: MLIR.verify(module), else: {:ok, module}

      case verified do
        {:ok, _} ->
          {:ok, module, []}

        {:error, diagnostics} ->
          MLIR.Module.destroy(module)
          {:error, diagnostics}
      end
    rescue
      e ->
        MLIR.Module.destroy(module)
        reraise e, __STACKTRACE__
    end
  end

  # Keep diagnostics and raising behavior identical across all three builder surfaces.
  for name <- [:build, :build_function, :build_module] do
    @doc "Calls `#{name}/3`, returning its result or raising with MLIR diagnostics."
    def unquote(:"#{name}!")(target, spec, opts \\ []),
      do: unwrap_build!(unquote(name)(target, spec, opts), unquote(name))
  end

  defp unwrap_build!({:ok, ops, results, _diagnostics}, _name), do: {ops, results}
  defp unwrap_build!({:ok, result, _diagnostics}, _name), do: result

  defp unwrap_build!({:error, diagnostics}, name),
    do: raise(ArgumentError, MLIR.Diagnostic.format(diagnostics, "IRBatch.#{name}! verification failed"))

  @doc "Maps function and block argument types and expands body steps with their `{block, step}` positions."
  def map_function(spec, type, expand) when is_function(type, 1) and is_function(expand, 2),
    do: __MODULE__.Spec.normalize(spec, type: type, expand: expand)

  defp resolve_container_block!(%MLIR.Block{} = block), do: block
  defp resolve_container_block!(%MLIR.Module{} = module), do: MLIR.Module.body(module)

  defp resolve_container_block!(%MLIR.Operation{} = op),
    do: MLIR.Module.body(op) || raise(ArgumentError, "target operation does not have an entry block")

  defp resolve_container_block!(other),
    do: raise(ArgumentError, "invalid build_function target: #{inspect(other)}")

  defp resolve_container_context!(%mod{} = op, _blk, _opts) when mod in [MLIR.Module, MLIR.Operation],
    do: MLIR.context(op)

  defp resolve_container_context!(_target, block, opts), do: resolve_build_context!(block, :end, [], opts)
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

  defp create_function_shell!(ctx, loc, name, symbol, inputs, results, spec) do
    visibility = spec[:visibility]

    unless visibility in [nil, :public, :private, :nested],
      do: raise(ArgumentError, "invalid function :visibility #{inspect(visibility)}")

    attrs = [sym_name: MLIR.Attribute.string(symbol, ctx: ctx)]

    attrs =
      if Map.get(spec, :include_function_type, name in ["func.func", "llvm.func"]) do
        type = build_function_type!(ctx, name, inputs, results, spec[:function_type])
        Keyword.put(attrs, :function_type, MLIR.Attribute.type(type))
      else
        attrs
      end

    attrs =
      if name == "func.func" and visibility in [:private, :nested],
        do: Keyword.put(attrs, :sym_visibility, MLIR.Attribute.string(to_string(visibility), ctx: ctx)),
        else: attrs

    extra = Enum.map(Map.get(spec, :attributes, []), fn {k, v} -> {String.to_atom(to_string(k)), v} end)

    op =
      %Beaver.Changeset{name: name, location: loc, context: ctx}
      |> Beaver.Changeset.add_argument(attrs ++ extra)
      |> Beaver.Changeset.add_argument(MLIR.Region.create())
      |> MLIR.Operation.create()

    if name != "func.func" and visibility in [:private, :nested], do: MLIR.SymbolTable.set_visibility(op, visibility)
    op
  end

  defp build_function_type!(ctx, _op_name, _inputs, _results, explicit) when is_binary(explicit),
    do: MLIR.Type.get(explicit, ctx: ctx)

  defp build_function_type!(_ctx, _op_name, _inputs, _results, %MLIR.Type{} = explicit), do: explicit

  defp build_function_type!(ctx, "llvm.func", input_types, result_types, _) do
    in_str = Enum.map_join(input_types, ", ", &to_string/1)

    ret_str =
      case result_types do
        [] -> "!llvm.void"
        [single] -> to_string(single)
        many -> "!llvm.struct<(#{Enum.map_join(many, ", ", &to_string/1)})>"
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

  defp normalize_function_blocks!(spec, input_names, input_types) do
    blocks =
      case Map.take(spec, [:blocks, :body]) do
        %{blocks: _, body: _} -> raise ArgumentError, "function spec accepts either :body or :blocks, not both"
        %{blocks: []} -> raise ArgumentError, ":blocks cannot be empty"
        %{blocks: blocks} -> blocks
        body -> [Map.put(body, :name, :entry)]
      end

    Enum.with_index(blocks, &normalize_block_entry!(&1, &2, input_names, input_types))
  end

  defp normalize_block_entry!({name, {args, body}}, idx, names, types)
       when (is_atom(name) or is_binary(name)) and is_list(args) and is_list(body) do
    map = %{name: name, body: body}
    map = if idx == 0 and args == [] and types != [], do: map, else: Map.put(map, :args, args)
    normalize_block_entry!(map, idx, names, types)
  end

  defp normalize_block_entry!(spec, idx, names, types) do
    spec = __MODULE__.Spec.block(spec)

    {names, types} =
      if idx == 0 and not Map.has_key?(spec, :args),
        do: {names, types},
        else: split_typed_args!(Map.get(spec, :args, []))

    %{name: Map.get(spec, :name, :"bb#{idx}"), arg_names: names, arg_types: types, body: Map.get(spec, :body, [])}
  end

  defp create_cfg_blocks!(region, specs, ctx, loc) do
    names = Enum.map(specs, & &1.name)
    if Enum.uniq(names) != names, do: raise(ArgumentError, "duplicate block name in IRBatch.build_function")

    Enum.with_index(specs, fn spec, index ->
      block =
        MLIR.Region.append_block(region, Enum.map(spec.arg_types, &resolve_mlir_type(&1, ctx)), ctx: ctx, location: loc)

      bindings =
        for {name, i} <- Enum.with_index(spec.arg_names),
            name not in [nil, :_],
            into: %{},
            do: {name, {:cfg_block_arg, index, i}}

      Map.merge(spec, %{block: block, index: index, bindings: bindings})
    end)
  end

  defp encode_cfg_blocks!(plan, blocks, return_op, loc, initial) do
    {encoded, {_, bindings}} =
      Enum.map_reduce(plan, {0, initial}, fn block, {index, bindings} ->
        bindings = Map.merge(bindings, block.bindings)

        Enum.map_reduce(block.body, {index, bindings}, fn stmt, {i, bindings} ->
          stmt = expand_cfg_statement!(stmt, blocks, return_op)
          {encoded, bindings} = encode_build_spec!(stmt, i, block.index, loc, bindings)
          {encoded, {i + 1, bindings}}
        end)
      end)

    {List.flatten(encoded), bindings}
  end

  defp expand_cfg_statement!(:return, _blocks, return_op), do: %{name: return_op, operands: []}

  defp expand_cfg_statement!({:return, operands}, _blocks, return_op),
    do: %{name: return_op, operands: List.wrap(operands)}

  defp expand_cfg_statement!({:br, dest_spec}, blocks, _return_op) do
    {blk, args} = normalize_branch_dest!(dest_spec, blocks)
    %{name: "cf.br", operands: args, successors: [blk]}
  end

  defp expand_cfg_statement!({:br, target, args}, blocks, _return_op) when is_list(args),
    do: %{name: "cf.br", operands: args, successors: [lookup_cfg_block!(target, blocks)]}

  defp expand_cfg_statement!({:cond_br, cond_ref, true_dest, false_dest}, blocks, _return_op) do
    {t_blk, t_args} = normalize_branch_dest!(true_dest, blocks)
    {f_blk, f_args} = normalize_branch_dest!(false_dest, blocks)

    %{
      name: "cf.cond_br",
      operands: [cond_ref] ++ t_args ++ f_args,
      successors: [t_blk, f_blk],
      attributes: [operandSegmentSizes: {:dense_i32, [1, length(t_args), length(f_args)]}]
    }
  end

  defp expand_cfg_statement!({:switch, {flag_type, flag_ref}, default_dest, cases}, blocks, return_op)
       when is_list(cases),
       do: expand_cfg_statement!({:switch, flag_ref, flag_type, default_dest, cases}, blocks, return_op)

  defp expand_cfg_statement!({:switch, flag_ref, flag_type, default_dest, cases}, blocks, _return_op)
       when is_list(cases) do
    {def_blk, def_args} = normalize_branch_dest!(default_dest, blocks)
    flag_type_str = normalize_type_string!(flag_type)
    {values, destinations} = Enum.map(cases, &switch_case!/1) |> Enum.unzip()
    {case_blocks, arguments} = Enum.map(destinations, &normalize_branch_dest!(&1, blocks)) |> Enum.unzip()
    operands = List.flatten(arguments)

    attrs = [
      operandSegmentSizes: {:dense_i32, [1, length(def_args), length(operands)]},
      case_operand_segments: {:dense_i32, Enum.map(arguments, &length/1)}
    ]

    case_values = "dense<[#{Enum.join(values, ", ")}]> : vector<#{length(values)}x#{flag_type_str}>"
    attrs = if values == [], do: attrs, else: Keyword.put(attrs, :case_values, {:parse, case_values})

    %{
      name: "cf.switch",
      operands: [flag_ref] ++ def_args ++ operands,
      successors: [def_blk | case_blocks],
      attributes: attrs
    }
  end

  defp expand_cfg_statement!(spec, blocks, _return_op) when is_map(spec) or is_list(spec) do
    map = Map.new(spec)

    if Map.has_key?(map, :successors) do
      Map.update!(map, :successors, fn succs -> Enum.map(succs, &lookup_cfg_block!(&1, blocks)) end)
    else
      map
    end
  end

  defp expand_cfg_statement!(other, _blocks, _return_op),
    do: raise(ArgumentError, "unsupported CFG statement in IRBatch.build_function: #{inspect(other)}")

  defp switch_case!({value, dest}) when is_integer(value), do: {value, dest}
  defp switch_case!({value, dest, args}) when is_integer(value) and is_list(args), do: {value, {dest, args}}
  defp switch_case!(other), do: raise(ArgumentError, "invalid :switch case specification: #{inspect(other)}")

  defp normalize_branch_dest!({dest, args}, blocks) when is_list(args),
    do: {lookup_cfg_block!(dest, blocks), args}

  defp normalize_branch_dest!(dest, blocks), do: {lookup_cfg_block!(dest, blocks), []}
  defp lookup_cfg_block!(%MLIR.Block{} = block, _blocks), do: block

  defp lookup_cfg_block!(name, blocks),
    do:
      Map.get(blocks, name) ||
        raise(ArgumentError, "unknown CFG block #{inspect(name)} (known: #{inspect(Map.keys(blocks))})")

  defp encode_verify_root(nil), do: nil
  defp encode_verify_root(false), do: nil
  defp encode_verify_root(root), do: MLIR.Operation.from_module(root).ref
  defp resolve_build_target!(%MLIR.Block{} = block, pos) when pos in [:end, :start], do: {block, pos}

  defp resolve_build_target!(%MLIR.Block{} = block, {rel, %MLIR.Operation{ref: op_ref}})
       when rel in [:before, :after], do: {block, {rel, op_ref}}

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

      is_tuple(position) ->
        MLIR.context(%MLIR.Operation{ref: elem(position, 1)})

      parent = MLIR.Block.parent_operation(block) ->
        MLIR.context(parent)

      true ->
        Enum.find_value(specs, &spec_context(Map.new(&1))) ||
          raise(ArgumentError, "unable to infer MLIR.Context for IRBatch.build; pass :ctx in opts")
    end
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
      __MODULE__.Protocol.operation(%{
        name: name,
        location: loc_ref,
        operands: operands,
        result_types: result_types,
        attributes: attributes,
        successors: successors,
        regions: regions,
        block: block_idx
      })

    {encoded, next_bindings}
  end

  defp encode_operand_ref!(%MLIR.Value{ref: ref}, _bindings), do: __MODULE__.Protocol.operand(:value, 0, ref)

  for {tag, kind} <- [op: :result, block_arg: :block_arg] do
    defp encode_operand_ref!({unquote(tag), index}, _) when is_integer(index) and index >= 0,
      do: __MODULE__.Protocol.operand(unquote(kind), index, 0)
  end

  for {tag, kind} <- [result: :result, block_arg: :cfg_block_arg] do
    defp encode_operand_ref!({unquote(tag), first, second}, _)
         when is_integer(first) and first >= 0 and is_integer(second) and second >= 0,
         do: __MODULE__.Protocol.operand(unquote(kind), first, second)
  end

  defp encode_operand_ref!(sym, bindings) when is_atom(sym) or is_binary(sym) do
    case Map.fetch(bindings, sym) do
      {:ok, {:cfg_block_arg, blk_idx, arg_idx}} -> __MODULE__.Protocol.operand(:cfg_block_arg, blk_idx, arg_idx)
      {:ok, {:op_result, op_idx, res_idx}} -> __MODULE__.Protocol.operand(:result, op_idx, res_idx)
      {:ok, %MLIR.Value{ref: ref}} -> __MODULE__.Protocol.operand(:value, 0, ref)
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
  defp normalize_type_string!(t), do: to_string(normalize_type_spec!(t))
  defp encode_type_term!(%MLIR.Type{ref: ref}), do: ref
  defp encode_type_term!(t), do: normalize_type_spec!(t)
  defp encode_attribute_pair!({attr_name, value}), do: {to_string(attr_name), encode_attribute_value!(value)}

  defp encode_attribute_pair!(other),
    do: raise(ArgumentError, "unsupported IRBatch.build attribute: #{inspect(other)}")

  defp encode_attribute_value!(%MLIR.Attribute{ref: ref}), do: ref
  defp encode_attribute_value!(:unit), do: :unit
  defp encode_attribute_value!(b) when is_boolean(b), do: b
  defp encode_attribute_value!(s) when is_binary(s) and byte_size(s) > 0, do: s
  defp encode_attribute_value!({:unit}), do: :unit
  defp encode_attribute_value!({tag, n}) when tag in [:i8, :i16, :i32, :i64, :index] and is_integer(n), do: {tag, n}
  defp encode_attribute_value!({:int, w, n}) when is_integer(w) and w > 0 and is_integer(n), do: {:int, w, n}
  defp encode_attribute_value!({tag, v}) when tag in [:bool, :i1] and (is_boolean(v) or v in [0, 1]), do: {tag, v}
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
