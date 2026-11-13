defmodule Beaver.SSA.Schema do
  @moduledoc """
  Declares SSA uses, definitions and edges using paths of map keys, list indices and `:each`.
  Exit variants generate traversal clauses; unknown exits have no values or edges. Opaque metadata and literals
  are preserved. Language policy, including effects, unwind and safepoints, remains with the caller.
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

      @instruction_paths for {_, key} <- @graph_schema.instruction, do: [key, :each]
      @block_paths for path <- @instruction_paths, do: [@graph_schema.block[:instructions], :each | path]
      @value_paths [[@graph_schema.block[:parameters], :each] | @block_paths]
      for role <- [:uses, :defs] do
        key = @graph_schema.instruction[role]
        paths = [[@graph_schema.block[:instructions], :each, key, :each]]
        paths = if role == :defs, do: [[@graph_schema.block[:parameters], :each] | paths], else: paths
        def unquote(:"instruction_#{role}")(node), do: Map.fetch!(node, unquote(key))

        def unquote(:"block_#{role}")(node) do
          exit = node[@graph_schema.block[:exit]]
          tail = if unquote(role) == :uses, do: exit_uses(exit), else: exit_values(exit, :defs)
          Beaver.SSA.Schema.values(node, unquote(Macro.escape(paths))) ++ tail
        end
      end

      def map_instruction_values(node, fun), do: Beaver.SSA.Schema.map_values(node, @instruction_paths, fun)

      def map_block_values(node, fun) do
        node
        |> Beaver.SSA.Schema.map_values(@value_paths, fun)
        |> Map.update!(@graph_schema.block[:exit], &map_exit_values(&1, fun))
      end
    end
  end

  @doc false
  def values(node, paths), do: Enum.flat_map(paths, &read(node, &1))
  @doc false
  def map_values(node, paths, fun), do: Enum.reduce(paths, node, &update(&2, &1, fun))
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
