defmodule ExLayersTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Ex
  alias Beaver.MLIR.Conversion.Ex.RuntimeABI
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Dialect.Ex, as: Dialect
  alias Beaver.MLIR.Operation

  @moduletag :smoke

  setup %{ctx: ctx} do
    assert Beaver.Slang.load(ctx, Dialect) |> MLIR.LogicalResult.success?()
    :ok
  end

  test "legacy operations have explicit owners and unknown boundaries reject" do
    assert Dialect.layer("ex.case") == :language
    assert Dialect.layer("ex.integer_mul") == :language
    assert Dialect.layer("ex.cont_save") == :execution
    assert Dialect.layer("ex.reduction_tick") == :execution
    assert Dialect.layer("ex.make_fun_with_signature") == :runtime
    assert Dialect.layer("ex.term_export") == :runtime
    assert length(Dialect.operations()) == 172

    assert Enum.flat_map(Dialect.layers(), &Dialect.operations/1) |> Enum.sort() ==
             Dialect.operations()

    assert_raise KeyError, fn -> Dialect.layer("ex.future") end

    for layers <- [[], [:future], [:language, :language]] do
      assert_raise ArgumentError, fn -> Ex.plan(layers: layers) end
    end
  end

  test "boundary checks find runtime operations inside nested function regions", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        ~S"""
        module {
          func.func @host() {
            %runtime = "ex.runtime_create"() : () -> i64
            %closed = "ex.runtime_destroy"(%runtime) : (i64) -> i64
            return
          }
        }
        """,
        ctx: ctx
      )

    assert Dialect.verify_layers!(module, :runtime) == module

    assert_raise ArgumentError, ~r/ex.runtime_create belongs to runtime/, fn ->
      Dialect.verify_layers!(module, [:language, :execution])
    end

    assert_raise ArgumentError, fn -> Dialect.verify_layers!(module, []) end
    assert_raise ArgumentError, fn -> Dialect.verify_layers!(module, :future) end
  end

  test "selected layers lower without consuming another owner's operations", %{ctx: ctx} do
    source = ~S"""
    module {
      func.func @step(%x: i64) -> i64 {
        %sum = "ex.add"(%x, %x) : (i64, i64) -> i64
        %tick = "ex.reduction_tick"(%sum) : (i64) -> i64
        %runtime = "ex.runtime_create"() : () -> i64
        %closed = "ex.runtime_destroy"(%runtime) : (i64) -> i64
        return %tick : i64
      }
    }
    """

    staged = MLIR.Module.create!(source, ctx: ctx) |> MLIR.verify!()
    staged = Plan.run!(Ex.plan(layers: :execution), staged) |> MLIR.verify!()
    names = names(staged)
    refute "ex.reduction_tick" in names
    assert "ex.add" in names
    assert "ex.runtime_create" in names
    assert Dialect.verify_layers!(staged, [:language, :runtime]) == staged

    staged = Plan.run!(Ex.plan(layers: :runtime), staged) |> MLIR.verify!()
    assert "ex.add" in names(staged)
    refute "ex.runtime_create" in names(staged)
    staged = Plan.run!(Ex.plan(layers: :language), staged) |> MLIR.verify!()
    refute Enum.any?(names(staged), &String.starts_with?(&1, "ex."))

    direct =
      MLIR.Module.create!(source, ctx: ctx) |> then(&Plan.run!(Ex.plan(), &1)) |> MLIR.verify!()

    assert functions(staged) == functions(direct)
  end

  test "pure scalar computation can disappear while runtime and exceptional work remain", %{
    ctx: ctx
  } do
    module =
      MLIR.Module.create!(
        ~S"""
        module {
          func.func @effects(%x: i64, %term: !ex.term) {
            %sum = "ex.add"(%x, %x) : (i64, i64) -> i64
            %division = "ex.div"(%x, %x) : (i64, i64) -> i64
            %integer = "ex.integer_mul"(%term, %term) : (!ex.term, !ex.term) -> !ex.term
            %sent = "ex.send"(%term, %term) : (!ex.term, !ex.term) -> !ex.term
            %runtime = "ex.runtime_create"() : () -> i64
            return
          }
        }
        """,
        ctx: ctx
      )
      |> MLIR.verify!()

    add = operations(module) |> Enum.find(&(Operation.name(&1) == "ex.add"))
    assert MLIR.MemoryEffects.get(add) == []
    assert MLIR.ConditionallySpeculatable.query(add) == :speculatable
    Beaver.Composer.append(module, MLIR.Transform.canonicalize()) |> Beaver.Composer.run!()
    refute "ex.add" in names(module)

    for name <- ~w(ex.div ex.integer_mul ex.send ex.runtime_create),
        do: assert(name in names(module))
  end

  test "runtime manifest retains aliases and specialized calling conventions", %{ctx: ctx} do
    assert RuntimeABI.intrinsics().reduction_tick == "ex.term.clock_tick"
    assert RuntimeABI.intrinsics().term_export == "ex.term.export"
    assert RuntimeABI.symbols() == Ex.term_intrinsic_symbols()
    assert signature("ex.term.cont_save", ctx) == "(i64, i64, i64) -> i64"
    assert signature("ex.term.worker_run", ctx) == "(i64, (i64) -> i64) -> i64"
    assert signature("ex.term.try_push", ctx) == "(!llvm.ptr) -> i64"
    assert_raise FunctionClauseError, fn -> RuntimeABI.function_type("ex.term.unknown", ctx) end
  end

  test "term values survive staged conversion across representation boundaries", %{ctx: ctx} do
    source = ~S"""
    module {
      "ex.func"() ({
      ^bb0(%x: i64):
        %word = "ex.box"(%x) : (i64) -> !ex.term
        %tuple = "ex.tuple"(%word) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> !ex.term
        %result = "ex.unbox"(%tuple) : (!ex.term) -> i64
        "ex.return"(%result) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      }) {sym_name = "pair"} : () -> ()
    }
    """

    direct =
      MLIR.Module.create!(source, ctx: ctx) |> then(&Plan.run!(Ex.plan(), &1)) |> MLIR.verify!()

    for order <- [[:runtime, :language], [:language, :runtime]] do
      module = MLIR.Module.create!(source, ctx: ctx) |> MLIR.verify!()

      staged =
        Enum.reduce(order, module, fn layer, ir -> Plan.run!(Ex.plan(layers: layer), ir) end)

      staged = Ex.finish!(staged) |> MLIR.verify!()
      assert functions(staged) == functions(direct)
    end
  end

  test "finishing rejects a type bridge without a representation-preserving inverse", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        ~S"""
        module {
          func.func @invalid_bridge(%x: f32) -> i64 {
            %word = "builtin.unrealized_conversion_cast"(%x) : (f32) -> i64
            return %word : i64
          }
        }
        """,
        ctx: ctx
      )

    assert_raise ArgumentError, ~r/unresolved Ex type bridge/, fn -> Ex.finish!(module) end
  end

  defp signature(symbol, ctx),
    do: RuntimeABI.function_type(symbol, ctx) |> Beaver.Deferred.resolve(ctx) |> MLIR.to_string()

  defp names(module), do: Enum.map(operations(module), &Operation.name/1)

  defp functions(module) do
    for op <- operations(module),
        Operation.name(op) == "func.func",
        into: %{},
        do: {MLIR.to_string(op["sym_name"]), MLIR.to_string(op, generic: true)}
  end

  defp operations(module) do
    Beaver.Walker.prewalk(module, [], fn
      %Operation{} = op, acc -> {op, [op | acc]}
      other, acc -> {other, acc}
    end)
    |> elem(1)
  end
end
