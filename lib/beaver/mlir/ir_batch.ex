defmodule Beaver.MLIR.IRBatch do
  @moduledoc """
  Bounded IR snapshots, edits, and block builders with one native submission per batch.
  """
  alias Beaver.{Execution, MLIR, Native}

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

  @doc "Builds and inserts a batch of up to 1024 operations into `target` in a single NIF submission."
  def build(target, specs, opts \\ []) when is_list(specs) and is_list(opts) do
    opts = Keyword.put_new(opts, :strategy, :auto)
    {block, position} = resolve_build_target!(target, Keyword.get(opts, :position, :end))

    case specs do
      [] ->
        {:ok, [], [], []}

      _ ->
        ctx = resolve_build_context!(block, position, specs, opts)
        default_loc = Keyword.get_lazy(opts, :location, fn -> MLIR.Location.unknown(ctx: ctx) end)

        verify_root_ref =
          case Keyword.get(opts, :verify) do
            nil -> nil
            false -> nil
            root -> MLIR.Operation.from_module(root).ref
          end

        encoded_specs = Enum.map(specs, &encode_build_spec!(&1, ctx, default_loc))

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

  defp resolve_build_target!(%MLIR.Block{} = block, pos) when pos in [:end, :start], do: {block, pos}

  defp resolve_build_target!(%MLIR.Block{} = block, {rel, %MLIR.Operation{ref: op_ref}})
       when rel in [:before, :after],
       do: {block, {rel, op_ref}}

  defp resolve_build_target!({rel, %MLIR.Operation{} = op}, _pos) when rel in [:before, :after] do
    block = MLIR.CAPI.mlirOperationGetBlock(op)
    if MLIR.null?(block), do: raise(ArgumentError, "anchor operation does not belong to a block")
    {block, {rel, op.ref}}
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
        parent_op = MLIR.CAPI.mlirBlockGetParentOperation(block)
        if MLIR.null?(parent_op), do: find_spec_context!(specs), else: MLIR.context(parent_op)
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
      _ -> nil
    end)
  end

  defp encode_build_spec!(spec, ctx, default_loc) when is_map(spec) or is_list(spec) do
    map = Map.new(spec)
    name = Map.fetch!(map, :name) |> to_string()
    %MLIR.Location{ref: loc_ref} = Map.get(map, :location, default_loc)

    operands =
      Enum.map(Map.get(map, :operands, []), fn
        %MLIR.Value{ref: ref} ->
          {0, 0, ref}

        {:op, op_idx} when is_integer(op_idx) and op_idx >= 0 ->
          {1, op_idx, 0}

        {:result, op_idx, res_idx} when is_integer(op_idx) and op_idx >= 0 and is_integer(res_idx) and res_idx >= 0 ->
          {1, op_idx, res_idx}

        {:block_arg, arg_idx} when is_integer(arg_idx) and arg_idx >= 0 ->
          {2, arg_idx, 0}

        other ->
          raise ArgumentError, "unsupported IRBatch.build operand reference: #{inspect(other)}"
      end)

    result_types =
      Enum.map(Map.get(map, :results, Map.get(map, :result_types, [])), fn
        %MLIR.Type{ref: ref} -> ref
        s when is_binary(s) -> MLIR.Type.get(s, ctx: ctx).ref
        other -> raise ArgumentError, "unsupported IRBatch.build result type: #{inspect(other)}"
      end)

    attributes =
      Enum.map(Map.get(map, :attributes, []), fn
        {attr_name, %MLIR.Attribute{ref: ref}} -> {to_string(attr_name), ref}
        {attr_name, s} when is_binary(s) -> {to_string(attr_name), MLIR.Attribute.get(s, ctx: ctx).ref}
        other -> raise ArgumentError, "unsupported IRBatch.build attribute: #{inspect(other)}"
      end)

    successors =
      Enum.map(Map.get(map, :successors, []), fn
        %MLIR.Block{ref: ref} -> ref
        other -> raise ArgumentError, "unsupported IRBatch.build successor: #{inspect(other)}"
      end)

    regions = Map.get(map, :regions, 0)
    unless is_integer(regions) and regions >= 0, do: raise(ArgumentError, ":regions must be a non-negative integer")
    {name, loc_ref, operands, result_types, attributes, successors, regions}
  end
end
