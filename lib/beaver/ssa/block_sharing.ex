defmodule Beaver.SSA.BlockSharing do
  @moduledoc """
  Shares alpha-equivalent blocks using caller-supplied eligibility, literal and jump policies.
  Entry blocks and escaping definitions are excluded. Complete normalized structures determine equality,
  preserving opaque metadata and instruction order; literals remain in the original predecessors.
  """

  def run(function, schema, opts) do
    fields = schema.schema()
    blocks = Map.fetch!(function, fields.function[:blocks])

    uses =
      Enum.group_by(
        for({block, index} <- Enum.with_index(blocks), value <- schema.block_uses(block), do: {value, index}),
        &elem(&1, 0),
        &elem(&1, 1)
      )

    candidates =
      for {block, index} <- Enum.with_index(blocks),
          index > 0,
          {arguments, normalized} <- List.wrap(candidate(block, index, uses, schema, opts)),
          do: {normalized, {index, arguments}}

    groups =
      Enum.group_by(candidates, &elem(&1, 0), &elem(&1, 1))
      |> Enum.filter(fn {_, members} -> length(members) >= 2 end)
      |> Enum.sort_by(fn {block, _} -> {Keyword.get(opts, :sort_key, &Function.identity/1).(block), block} end)

    initial = {:array.from_list(blocks), length(blocks), function[fields.function[:values]]}
    {blocks, _, values} = Enum.reduce(groups, initial, &rewrite_group(&1, &2, schema, opts))

    function
    |> Map.put(fields.function[:blocks], :array.to_list(blocks))
    |> Map.put(fields.function[:values], values)
  end

  defp rewrite_group({template, members}, {blocks, target, values}, schema, opts) do
    fields = schema.schema().block
    literal? = Keyword.fetch!(opts, :literal?)
    jump = Keyword.fetch!(opts, :jump)

    blocks =
      Enum.reduce(members, blocks, fn {index, arguments}, blocks ->
        block = :array.get(index, blocks)

        replacement =
          block
          |> Map.update!(fields[:instructions], &Enum.filter(&1, literal?))
          |> Map.put(fields[:exit], jump.(target, arguments))

        :array.set(index, replacement, blocks)
      end)

    shared = schema.map_block_values(template, &(&1 + values))
    {:array.set(target, shared, blocks), target + 1, values + length(schema.block_defs(template))}
  end

  defp candidate(block, index, uses, schema, opts) do
    fields = schema.schema().block
    definitions = MapSet.new(schema.block_defs(block))
    produced = MapSet.difference(definitions, MapSet.new(block[fields[:parameters]]))
    escapes? = Enum.any?(produced, fn value -> Enum.any?(Map.get(uses, value, []), &(&1 != index)) end)

    if Keyword.fetch!(opts, :eligible?).(block) and not escapes? do
      {literals, instructions} = Enum.split_with(block[fields[:instructions]], Keyword.fetch!(opts, :literal?))
      seeds = Enum.flat_map(literals, &schema.instruction_defs/1)

      arguments =
        schema.block_uses(block)
        |> Enum.reject(&MapSet.member?(produced, &1))
        |> then(&Enum.uniq(seeds ++ &1))

      if length(arguments) <= Keyword.fetch!(opts, :max_arguments) and
           length(instructions) >= Keyword.get(opts, :min_instructions, 1) do
        template = block |> Map.put(fields[:parameters], arguments) |> Map.put(fields[:instructions], instructions)
        {arguments, alpha_normalize(template, arguments, schema)}
      end
    end
  end

  defp alpha_normalize(template, arguments, schema) do
    names = (arguments ++ schema.block_defs(template)) |> Enum.uniq() |> Enum.with_index() |> Map.new()
    schema.map_block_values(template, &Map.fetch!(names, &1))
  end
end
