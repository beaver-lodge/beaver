defmodule Beaver.IRBatchSpecTest do
  use Beaver.Case, async: true
  alias Beaver.{Execution, MLIR}
  alias Beaver.MLIR.{IRBatch, Location, Module, Operation}
  alias Beaver.MLIR.IRBatch.{Builder, CFG, Protocol, Spec}
  import CFG, only: [cfg: 2]

  defmodule Language do
    use Spec, types: [term: :i64], calls: [{:pair, "pair", 2, [:term, :term]}]
  end

  test "declarations normalize tuples and keywords, preserving names and multivalue bindings", %{ctx: ctx} do
    spec =
      Language.normalize_function(
        {:func, "identity", [term: :term], :term,
         blocks: [{:entry, [{:br, :done, [:term]}]}, {:done, [value: :term], [{:return, [:value]}]}]}
      )

    assert spec.inputs == [term: :i64]
    assert spec.results == [:i64]
    assert List.last(spec.blocks).args == [value: :i64]
    spec = Map.put(spec, :bindings, term: {:cfg_block_arg, 0, 99})
    Module.with_module(IRBatch.build_module!(ctx, [spec]), &MLIR.verify!/1)

    assert [%{bind: [:a, :b], operands: [:x, :y], results: [:i64, :i64]}] =
             Language.expand_declared_operation({:pair, {:a, :b}, :x, :y})

    normalized = Language.normalize_function(name: "pairs", body: [{:pair, {:a, :b}, :x, :y}])
    assert [%{name: "func.call", results: [:i64, :i64]}] = normalized.body

    assert :unknown = Language.expand_declared_operation({:pair, :a, :x})
    assert_raise ArgumentError, fn -> Spec.normalize(%{body: [], blocks: []}) end
  end

  test "CFG binds one emission, threads runtime arms and leaves caller variables hygienic", %{ctx: ctx} do
    builder = :untouched
    owner = self()

    compiled =
      cfg Builder.new() do
        block :entry do
          value <-
            (
              send(owner, :evaluated)
              Builder.expression(:binary, ["addi", [:x, :x], :i64])
            )

          arms = [{0, :zero, value}, {1, :one, value}]

          dispatch(:selector, :i32, :default, arms, fn operand, state ->
            Builder.end_block(state, {:return, [operand]})
          end)
        end

        block :default do
          terminate({:return, [:x]})
        end
      end

    assert builder == :untouched
    assert_received :evaluated
    refute_received :evaluated
    spec = %{name: "dispatch", inputs: [x: :i64, selector: :i32], results: [:i64], blocks: Builder.finish(compiled)}
    Module.with_module(IRBatch.build_module!(ctx, [spec]), &MLIR.verify!/1)

    assert_raise ArgumentError, ~r/duplicate CFG block/, fn ->
      Macro.expand_once(
        quote(
          do:
            CFG.cfg Builder.new() do
              block :same do
                terminate({:return, []})
              end

              block :same do
                terminate({:return, []})
              end
            end
        ),
        __ENV__
      )
    end
  end

  test "loaded native schema agrees and malformed batches fail before mutation", %{ctx: ctx} do
    assert Protocol.native_schema(ctx) == Protocol.schema()

    Module.with_module(ctx, "module {}", fn module ->
      block = Module.body(module)
      root = Operation.from_module(module)
      location = Location.unknown(ctx: ctx)

      fields = %{
        name: "arith.constant",
        location: location.ref,
        operands: [],
        result_types: ["i64"],
        attributes: [{"value", {:i64, 1}}],
        successors: [],
        regions: 0,
        block: nil
      }

      valid = Protocol.operation(fields)
      before = MLIR.to_string(module)

      invalid = [
        List.to_tuple(Tuple.to_list(valid) ++ [0, 0]),
        Protocol.operation(%{fields | block: 999}),
        Protocol.operation(%{fields | operands: [{99, 0, 0}]}),
        Protocol.operation(%{fields | operands: [Protocol.operand(:result, 4, 0)]}),
        Protocol.operation(%{fields | operands: [Protocol.operand(:block_arg, 8, 0)]}),
        Protocol.operation(%{fields | operands: [Protocol.operand(:cfg_block_arg, 7, 0)]})
      ]

      for bad <- invalid do
        assert_raise ArgumentError, fn ->
          Execution.run(ctx, :build, {block.ref, :end, [valid, bad], root.ref}, strategy: :auto)
        end

        assert MLIR.to_string(module) == before
      end
    end)
  end
end
