defmodule Beaver.MLIR.IRBatch.Spec do
  @moduledoc """
  Function/block normalization with declarative operation shortcuts.
  `use Spec, types: [...], calls: [{shortcut, symbol, arity, results}]` generates aliases and call expansion.
  Custom expansion callbacks handle literals or variable arity; symbolic names are never mapped as types.
  """

  @types Map.new(~w(i1 i8 i16 i32 i64 i128 index f16 bf16 f32 f64 none)a, &{&1, to_string(&1)})
         |> Map.put(:ptr, "!llvm.ptr")
  @doc "Builtin shorthand types shared by deferred changesets and batch specifications."
  def types, do: @types

  defmacro __using__(opts) do
    quote bind_quoted: [opts: opts] do
      for {name, type} <- Keyword.get(opts, :types, []) do
        def spec_type(unquote(name)), do: unquote(type)
      end

      def spec_type(type), do: type
      @spec_calls Keyword.get(opts, :calls, [])
      if length(Enum.uniq_by(@spec_calls, &elem(&1, 0))) != length(@spec_calls),
        do: raise(ArgumentError, "duplicate operation shortcut")

      for {name, symbol, arity, results} <- @spec_calls do
        unless is_atom(name) and is_binary(symbol) and is_integer(arity) and arity >= 0 and is_list(results),
          do: raise(ArgumentError, "invalid call specification")

        def expand_declared_operation(tuple)
            when is_tuple(tuple) and tuple_size(tuple) == unquote(arity + 2) and elem(tuple, 0) == unquote(name) do
          [_, bind | operands] = Tuple.to_list(tuple)
          [Beaver.MLIR.IRBatch.Spec.call(unquote(symbol), operands, Enum.map(unquote(results), &spec_type/1), bind)]
        end
      end

      def expand_declared_operation(_), do: :unknown

      def normalize_function(raw, expand \\ &expand_default/2),
        do: Beaver.MLIR.IRBatch.Spec.normalize(raw, type: &spec_type/1, expand: expand, attribute_aliases: [:at])

      defp expand_default(step, _) do
        case expand_declared_operation(step) do
          :unknown -> [step]
          operations -> operations
        end
      end
    end
  end

  def call(symbol, operands, results, bind) do
    bind = if is_tuple(bind), do: Tuple.to_list(bind), else: bind
    %{name: "func.call", operands: operands, results: results, attributes: [callee: {:symbol, symbol}], bind: bind}
  end

  @doc "Normalizes maps, keyword lists and function/block tuples without allocating native resources."
  def normalize(raw, opts \\ []) do
    type = Keyword.get(opts, :type, &Function.identity/1)
    expand = Keyword.get(opts, :expand, fn step, _ -> [step] end)
    spec = function(raw)
    spec = Map.put(spec, :kind, spec[:kind] || :func)
    aliases = Keyword.get(opts, :attribute_aliases, [])
    attributes = Enum.flat_map([:attributes | aliases], &(spec[&1] || []))
    spec = Map.put(spec, :attributes, attributes)

    spec =
      spec
      |> map_present(:inputs, &arguments(&1, type))
      |> map_present(:results, &Enum.map(List.wrap(&1), type))

    normalize_body(spec, type, expand, opts)
  end

  defp normalize_body(spec, type, expand, opts) do
    spec = if opts[:empty_blocks_as_body] && spec[:blocks] in [nil, []], do: Map.delete(spec, :blocks), else: spec

    if Map.has_key?(spec, :blocks) do
      if spec[:body], do: raise(ArgumentError, "function cannot contain both :body and :blocks")

      blocks =
        Enum.with_index(spec.blocks, fn raw, index ->
          raw |> block() |> map_present(:args, &arguments(&1, type)) |> expand_body(index, expand)
        end)

      Map.put(spec, :blocks, blocks)
    else
      expand_body(spec, 0, expand)
    end
  end

  defp expand_body(spec, block, expand) do
    body = Enum.with_index(spec[:body] || []) |> Enum.flat_map(fn {step, index} -> expand.(step, {block, index}) end)
    Map.put(spec, :body, body)
  end

  def function(%{} = spec), do: spec
  def function(spec) when is_list(spec), do: Map.new(spec)
  def function({kind, name, inputs, results}), do: function({kind, name, inputs, results, []})

  def function({kind, name, inputs, results, extra}) when is_list(extra) or is_map(extra),
    do: extra |> Map.new() |> Map.merge(%{kind: kind, name: name, inputs: inputs, results: results})

  def function(other), do: raise(ArgumentError, "invalid function specification: #{inspect(other)}")

  def block(%{} = spec), do: spec
  def block(spec) when is_list(spec), do: Map.new(spec)
  def block({name, body}), do: %{name: name, body: body}
  def block({name, args, body}), do: %{name: name, args: args, body: body}
  def block(other), do: raise(ArgumentError, "invalid block specification: #{inspect(other)}")

  defp arguments(values, type) do
    Enum.map(values, fn
      {name, value} -> {name, type.(value)}
      value -> type.(value)
    end)
  end

  defp map_present(map, key, fun), do: if(Map.has_key?(map, key), do: Map.update!(map, key, fun), else: map)
end
