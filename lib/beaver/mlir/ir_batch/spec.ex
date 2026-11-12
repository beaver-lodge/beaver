defmodule Beaver.MLIR.IRBatch.Spec do
  @moduledoc """
  Function/block normalization and declarative operation shortcuts.

  `use Spec, types: [...], calls: [{shortcut, symbol, arity, results}]` generates
  type alias clauses and fixed-arity call expansion. A language can add its own
  expansion callback for literal injection or variable-arity operations.
  Symbolic argument names are never treated as type aliases.
  """

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

      def normalize_function(raw) do
        normalize_function(raw, fn step, _ ->
          case expand_declared_operation(step) do
            :unknown -> [step]
            operations -> operations
          end
        end)
      end

      def normalize_function(raw, expand) do
        Beaver.MLIR.IRBatch.Spec.normalize(raw, type: &spec_type/1, expand: expand, attribute_aliases: [:at])
      end
    end
  end

  def call(symbol, operands, results, bind) do
    %{
      name: "func.call",
      operands: operands,
      results: results,
      attributes: [callee: {:symbol, symbol}],
      bind: if(is_tuple(bind), do: Tuple.to_list(bind), else: bind)
    }
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
    cond do
      not Map.has_key?(spec, :blocks) ->
        Map.put(spec, :body, body(spec[:body] || [], 0, expand))

      Keyword.get(opts, :empty_blocks_as_body, false) and spec.blocks in [nil, []] ->
        spec |> Map.delete(:blocks) |> Map.put(:body, body(spec[:body] || [], 0, expand))

      true ->
        if spec[:body], do: raise(ArgumentError, "function cannot contain both :body and :blocks")

        Map.put(
          spec,
          :blocks,
          Enum.with_index(spec.blocks, fn raw, index ->
            block = block(raw) |> map_present(:args, &arguments(&1, type))
            Map.put(block, :body, body(block[:body] || [], index, expand))
          end)
        )
    end
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

  defp body(steps, block, expand),
    do: steps |> Enum.with_index() |> Enum.flat_map(fn {step, index} -> expand.(step, {block, index}) end)

  defp map_present(map, key, fun), do: if(Map.has_key?(map, key), do: Map.update!(map, key, fun), else: map)
end
