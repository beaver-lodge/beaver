defmodule Beaver.SSA.Schema do
  @moduledoc """
  Declares value and edge positions in a caller-owned SSA representation.

  Paths contain map keys, list indices, and `:each` for repeated fields. Exit
  variants generate traversal clauses; opaque metadata and non-value literals
  are preserved. Unknown exits have no values or edges. Language policy (effects,
  unwind behavior and safepoints) belongs in the caller, not in the schema.
  """

  defmacro __using__(opts) do
    quote bind_quoted: [opts: opts] do
      @graph_schema Map.new(opts)
      def schema, do: @graph_schema

      for {tag, roles} <- Keyword.fetch!(opts, :exits) do
        def exit_values(%{unquote(tag) => node}, role) do
          paths = Keyword.get(unquote(Macro.escape(roles)), role, [])
          Enum.flat_map(paths, &Beaver.SSA.Schema.read(node, &1))
        end

        def map_exit_role(%{unquote(tag) => node} = exit, role, fun) do
          paths = Keyword.get(unquote(Macro.escape(roles)), role, [])
          Map.put(exit, unquote(tag), Enum.reduce(paths, node, &Beaver.SSA.Schema.update(&2, &1, fun)))
        end
      end

      def exit_values(_, _), do: []
      def map_exit_role(exit, _, _), do: exit
      def edges(exit), do: exit_values(exit, :edges)
      def map_edges(exit, fun), do: map_exit_role(exit, :edges, fun)

      def exit_uses(exit) do
        arguments = @graph_schema.edge[:arguments]
        Enum.flat_map(edges(exit), &Map.fetch!(&1, arguments)) ++ exit_values(exit, :uses)
      end

      def map_exit_values(exit, fun) do
        arguments = @graph_schema.edge[:arguments]

        exit
        |> map_exit_role(:uses, fun)
        |> map_exit_role(:defs, fun)
        |> map_edges(&Map.update!(&1, arguments, fn values -> Enum.map(values, fun) end))
      end

      def instruction_uses(instruction), do: Map.fetch!(instruction, @graph_schema.instruction[:uses])
      def instruction_defs(instruction), do: Map.fetch!(instruction, @graph_schema.instruction[:defs])

      def map_instruction_values(instruction, fun) do
        Enum.reduce(@graph_schema.instruction, instruction, fn {_, key}, node ->
          Map.update!(node, key, &Enum.map(&1, fun))
        end)
      end

      def block_defs(block) do
        fields = @graph_schema.block

        block[fields[:parameters]] ++
          Enum.flat_map(block[fields[:instructions]], &instruction_defs/1) ++
          exit_values(block[fields[:exit]], :defs)
      end

      def block_uses(block) do
        fields = @graph_schema.block
        Enum.flat_map(block[fields[:instructions]], &instruction_uses/1) ++ exit_uses(block[fields[:exit]])
      end

      def map_block_values(block, fun) do
        fields = @graph_schema.block

        block
        |> Map.update!(fields[:parameters], &Enum.map(&1, fun))
        |> Map.update!(fields[:instructions], &Enum.map(&1, fn node -> map_instruction_values(node, fun) end))
        |> Map.update!(fields[:exit], &map_exit_values(&1, fun))
      end
    end
  end

  @doc false
  def read(value, []), do: [value]
  def read(values, [:each | rest]), do: Enum.flat_map(values, &read(&1, rest))
  def read(values, [index | rest]) when is_integer(index), do: read(Enum.fetch!(values, index), rest)
  def read(map, [key | rest]), do: read(Map.fetch!(map, key), rest)

  @doc false
  def update(value, [], fun), do: fun.(value)
  def update(values, [:each | rest], fun), do: Enum.map(values, &update(&1, rest, fun))

  def update(values, [index | rest], fun) when is_integer(index),
    do: List.update_at(values, index, &update(&1, rest, fun))

  def update(map, [key | rest], fun), do: Map.update!(map, key, &update(&1, rest, fun))
end
