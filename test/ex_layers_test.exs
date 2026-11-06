defmodule ExLayersTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Ex
  alias Beaver.MLIR.Conversion.Ex.CRuntime
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Dialect.Ex, as: Dialect
  alias Beaver.MLIR.Operation

  @moduletag :smoke

  setup %{ctx: ctx} do
    assert Beaver.Slang.load(ctx, Dialect) |> MLIR.LogicalResult.success?()
    :ok
  end

  test "C ABI operations have explicit owners and unknown boundaries reject" do
    assert Dialect.layer("ex.case") == :language
    assert Dialect.layer("ex.integer_mul") == :language
    assert Dialect.layer("ex.btt_step") == :execution
    assert Dialect.layer("ex.btt_call") == :execution
    assert Dialect.layer("ex.btt_create") == :runtime
    assert Dialect.layer("ex.btt_release") == :runtime

    for retired <- ~w(ex.cont_save ex.worker_run ex.runtime_enter ex.make_fun ex.to_word),
        do: assert_raise(KeyError, fn -> Dialect.layer(retired) end)

    assert Enum.flat_map(Dialect.layers(), &Dialect.operations/1) |> Enum.sort() ==
             Dialect.operations()

    assert_raise KeyError, fn -> Dialect.layer("ex.future") end

    for layers <- [[], [:future], [:language, :language]] do
      assert_raise ArgumentError, fn -> Ex.plan(layers: layers) end
    end
  end

  test "boundary checks find runtime operations inside nested function regions", %{ctx: ctx} do
    module =
      parse(ctx, ~S"""
      func.func @host(%abi: i64, %out: !llvm.ptr) {
        %status = "ex.btt_create"(%abi, %out) : (i64, !llvm.ptr) -> i32
        return
      }
      """)

    assert Dialect.verify_layers!(module, :runtime) == module

    assert_raise ArgumentError, ~r/ex.btt_create belongs to runtime/, fn ->
      Dialect.verify_layers!(module, [:language, :execution])
    end

    assert_raise ArgumentError, fn -> Dialect.verify_layers!(module, []) end
    assert_raise ArgumentError, fn -> Dialect.verify_layers!(module, :future) end
  end

  test "selected layers lower without consuming another owner's operations", %{ctx: ctx} do
    source = ~S"""
    func.func @step(%x: i64, %runtime: !llvm.ptr, %out: !llvm.ptr) -> i32 {
      %sum = "ex.add"(%x, %x) : (i64, i64) -> i64
      %one = arith.constant 1 : i32
      %tick = "ex.btt_step"(%runtime, %one, %out) : (!llvm.ptr, i32, !llvm.ptr) -> i32
      %query = "ex.btt_query"(%sum, %out) : (i64, !llvm.ptr) -> i32
      return %tick : i32
    }
    """

    staged = parse(ctx, source)

    staged =
      Plan.run!(Ex.plan(layers: :execution, runtime_namespace: "layers"), staged)
      |> MLIR.verify!()

    names = names(staged)
    refute "ex.btt_step" in names
    assert "ex.add" in names
    assert "ex.btt_query" in names
    assert Dialect.verify_layers!(staged, [:language, :runtime]) == staged

    staged =
      Plan.run!(Ex.plan(layers: :runtime, runtime_namespace: "layers"), staged) |> MLIR.verify!()

    assert "ex.add" in names(staged)
    refute "ex.btt_query" in names(staged)
    staged = Plan.run!(Ex.plan(layers: :language), staged) |> MLIR.verify!()
    refute Enum.any?(names(staged), &String.starts_with?(&1, "ex."))

    direct =
      parse(ctx, source)
      |> then(&Plan.run!(Ex.plan(runtime_namespace: "layers"), &1))
      |> MLIR.verify!()

    assert functions(staged) == functions(direct)
  end

  test "canonicalization removes pure scalars but preserves effects", %{ctx: ctx} do
    module =
      parse(ctx, ~S"""
      func.func @effects(%x: i64, %term: !ex.term, %out: !llvm.ptr) {
        %sum = "ex.add"(%x, %x) : (i64, i64) -> i64
        %division = "ex.div"(%x, %x) : (i64, i64) -> i64
        %integer = "ex.integer_mul"(%term, %term) : (!ex.term, !ex.term) -> !ex.term
        %sent = "ex.send"(%term, %term) : (!ex.term, !ex.term) -> !ex.term
        %runtime = "ex.btt_create"(%x, %out) : (i64, !llvm.ptr) -> i32
        return
      }
      """)

    add = operations(module) |> Enum.find(&(Operation.name(&1) == "ex.add"))
    assert MLIR.MemoryEffects.get(add) == []
    assert MLIR.ConditionallySpeculatable.query(add) == :speculatable
    Beaver.Composer.append(module, MLIR.Transform.canonicalize()) |> Beaver.Composer.run!()
    refute "ex.add" in names(module)

    for name <- ~w(ex.div ex.integer_mul ex.send ex.btt_create),
        do: assert(name in names(module))
  end

  test "C ABI aggregates cross the compiler boundary by address", %{ctx: ctx} do
    assert signature("step", ctx) == "(!llvm.ptr, i32, !llvm.ptr) -> i32"
    assert signature("reply", ctx) == "(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i32) -> i32"
    assert CRuntime.abi_id() == 0x425454000000000C
    assert CRuntime.status(:busy) == 3
    assert_raise KeyError, fn -> CRuntime.function_type("legacy") end

    assert_raise ArgumentError, ~r/abstract Ex terms/, fn ->
      Ex.convert_type(Beaver.MLIR.Type.get("!ex.term", ctx: ctx))
    end
  end

  test "finishing rejects a type bridge without a representation-preserving inverse", %{ctx: ctx} do
    module =
      parse(ctx, ~S"""
      func.func @invalid_bridge(%x: f32) -> i64 {
        %word = "builtin.unrealized_conversion_cast"(%x) : (f32) -> i64
        return %word : i64
      }
      """)

    assert_raise ArgumentError, ~r/unresolved Ex type bridge/, fn -> Ex.finish!(module) end
  end

  defp parse(ctx, body),
    do: MLIR.Module.create!("module {\n#{body}\n}", ctx: ctx) |> MLIR.verify!()

  defp signature(symbol, ctx),
    do: CRuntime.function_type(symbol) |> Beaver.Deferred.resolve(ctx) |> MLIR.to_string()

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
