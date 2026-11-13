defmodule Beaver.SSASchemaTest do
  use ExUnit.Case, async: true
  alias Beaver.SSA.{BlockSharing, Flow, Scalar}

  defmodule Graph do
    use Beaver.SSA.Schema,
      function: [blocks: :blocks, values: :values],
      block: [parameters: :parameters, instructions: :instructions, exit: :exit],
      instruction: [uses: :inputs, defs: :outputs],
      edge: [target: :block, arguments: :arguments],
      exits: [
        {:jump, [edges: [[]]]},
        {:return, [uses: [[]]]},
        {:call, [uses: [[:arguments, :each]], defs: [[:result]], edges: [[:next]]]},
        {:switch, [uses: [[:value]], edges: [[:cases, :each, 1], [:default]]]}
      ]
  end

  test "schema maps values and edges without rewriting literal keys or metadata" do
    edge = %{block: 9, arguments: [3]}
    exit = %{switch: %{value: 1, cases: [[42, edge]], default: edge}, note: "opaque"}
    assert Graph.exit_uses(exit) == [3, 3, 1]

    assert Graph.map_exit_values(exit, &(&1 + 10)) == %{
             switch: %{value: 11, cases: [[42, %{block: 9, arguments: [13]}]], default: %{block: 9, arguments: [13]}},
             note: "opaque"
           }

    assert Graph.map_edges(exit, &%{&1 | block: 7}).switch.cases == [[42, %{block: 7, arguments: [3]}]]
    call = %{call: %{arguments: [1, 2], result: 3, next: edge}}
    assert Graph.exit_values(call, :defs) == [3]

    assert Graph.map_exit_values(call, &(&1 * 2)).call == %{
             arguments: [2, 4],
             result: 6,
             next: %{block: 9, arguments: [6]}
           }

    assert Graph.map_exit_values(:unreachable, &(&1 + 1)) == :unreachable
  end

  test "sharing preserves literal-dependent results and effect order" do
    original = fixture()
    shared = BlockSharing.run(original, Graph, options())
    assert length(shared.blocks) == 4
    assert shared.values == 12
    assert length(Enum.at(shared.blocks, 1).instructions) == 1
    assert length(Enum.at(shared.blocks, 2).instructions) == 1

    for entry <- [1, 2], input <- [-8, 0, 11] do
      assert execute(original.blocks, entry, [input], []) == execute(shared.blocks, entry, [input], [])
    end
  end

  test "escaping definitions, unwind and unequal metadata prevent sharing even with equal sort keys" do
    original = fixture()
    [entry, first, second] = original.blocks
    escaping = %{original | blocks: [%{entry | exit: %{return: 3}}, first, second]}
    assert BlockSharing.run(escaping, Graph, options()) == escaping
    unwind = %{original | blocks: [entry, Map.put(first, :unwind, 0), second]}
    assert BlockSharing.run(unwind, Graph, options()) == unwind
    metadata = %{original | blocks: [entry, Map.put(first, :effect, :before), Map.put(second, :effect, :after)]}
    assert BlockSharing.run(metadata, Graph, Keyword.put(options(), :sort_key, fn _ -> :collision end)) == metadata
  end

  test "unknown opcodes remain unknown while named nil and integer operations carry facts" do
    vocabulary = %{"nil" => nil, "integer" => :integer}
    assert Scalar.instruction("external", [], 1, vocabulary) == [:any]
    assert Scalar.instruction("nil", [], 1, vocabulary) == [nil]
    assert Scalar.instruction(%{"integer" => 0}, [], 1, vocabulary) == [{:integer, 0, 0}]
    assert Scalar.infer("add", [{:integer, 2, 3}, {:integer, 5, 7}]) == {:integer, 7, 10}
  end

  test "graph cleanup and allocation preserve observable effects with atom-keyed schemas" do
    %{blocks: [entry, first, second]} = fixture()
    entry = %{entry | parameters: [8], exit: %{jump: %{block: 1, arguments: [8]}}}
    dead = %{op: 99, inputs: [], outputs: [9]}
    first = %{first | instructions: [dead | first.instructions]}
    original = %{blocks: [entry, first, second], values: 10}
    blocks = Flow.dead_code(original.blocks, Graph, &(&1.op != :trace))
    compact = Flow.prune(%{original | blocks: blocks}, Graph)
    assert compact.values == 5 and length(compact.blocks) == 2
    allocated = Flow.allocate(compact, Graph, 32)
    assert allocated["frame_size"] > 0
    assert length(allocated["layout"]) == compact.values

    for input <- [-8, 0, 11],
        do: assert(execute(original.blocks, 0, [input], []) == execute(allocated.blocks, 0, [input], []))
  end

  defp options do
    [
      eligible?: &(not Map.has_key?(&1, :unwind)),
      literal?: &is_integer(&1.op),
      max_arguments: 4,
      min_instructions: 2,
      jump: fn block, arguments -> %{jump: %{block: block, arguments: arguments}} end
    ]
  end

  defp fixture do
    blocks =
      for {offset, literal} <- [{0, 2}, {4, 7}] do
        %{
          parameters: [offset],
          instructions: [
            %{op: literal, inputs: [], outputs: [offset + 1]},
            %{op: :add, inputs: [offset, offset + 1], outputs: [offset + 2]},
            %{op: :trace, inputs: [offset + 2], outputs: [offset + 3]}
          ],
          exit: %{return: offset + 3}
        }
      end

    %{blocks: [%{parameters: [], instructions: [], exit: :unreachable} | blocks], values: 8}
  end

  defp execute(blocks, index, arguments, trace) do
    block = Enum.at(blocks, index)

    {values, trace} =
      Enum.reduce(block.instructions, {Map.new(Enum.zip(block.parameters, arguments)), trace}, fn instruction,
                                                                                                  {values, trace} ->
        operands = Enum.map(instruction.inputs, &Map.fetch!(values, &1))

        {value, trace} =
          case {instruction.op, operands} do
            {literal, []} when is_integer(literal) -> {literal, trace}
            {:add, [a, b]} -> {a + b, trace}
            {:trace, [value]} -> {value, trace ++ [value]}
          end

        {[output], value} = {instruction.outputs, value}
        {Map.put(values, output, value), trace}
      end)

    case block.exit do
      %{return: value} -> {Map.fetch!(values, value), trace}
      %{jump: edge} -> execute(blocks, edge.block, Enum.map(edge.arguments, &Map.fetch!(values, &1)), trace)
    end
  end
end
