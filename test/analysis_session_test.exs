defmodule Beaver.MLIR.AnalysisSessionTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias MLIR.{AnalysisSession, DataLayout, Operation, Type}

  setup %{ctx: ctx} do
    module =
      "module {\n  func.func @compute(%x: i32) -> i32 {\n    %a = arith.addi %x, %x : i32\n    %b = arith.muli %a, %x : i32\n    return %b : i32\n  }\n}\n"
      |> MLIR.Module.create!(ctx: ctx)

    on_exit(fn -> MLIR.Module.destroy(module) end)
    func = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)
    block = func |> Beaver.Walker.regions() |> Enum.at(0) |> Beaver.Walker.blocks() |> Enum.at(0)
    [add, mul, ret] = Enum.to_list(Beaver.Walker.operations(block))
    %{module: module, add: add, mul: mul, ret: ret, arg: MLIR.Block.get_arg!(block, 0)}
  end

  test "caches batched analyses and invalidates after a batch mutation", data do
    %{module: module, add: add, mul: mul, ret: ret, arg: arg} = data

    AnalysisSession.with(module, fn session ->
      value = Operation.result(add, 0)

      assert %{generation: 0, cached: [:dominance, :post_dominance, :liveness], results: [true, true, false]} =
               AnalysisSession.query(session, [
                 {:dominates, add, mul},
                 {:post_dominates, ret, add},
                 {:dead_after, add, value}
               ])

      assert %{generation: 0, results: [true]} = AnalysisSession.query(session, [{:dominates, add, mul}])
      assert {:ok, _, _} = MLIR.IRBatch.update(module, [{:operand, mul, 0, arg}], analysis_session: session)
      assert %{generation: 2, cached: []} = AnalysisSession.stats(session)
      assert AnalysisSession.dead_after?(session, value, add)
    end)
  end

  test "preservation is explicit and a failed mutation still clears cached results", %{
    module: module,
    add: add,
    mul: mul
  } do
    AnalysisSession.with(module, fn session ->
      assert AnalysisSession.dominates?(session, add, mul)
      assert AnalysisSession.post_dominates?(session, mul, add)
      assert 1 = AnalysisSession.invalidate(session, [:dominance])
      assert %{generation: 1, cached: [:dominance]} = AnalysisSession.stats(session)

      assert_raise RuntimeError, "mutation failed", fn ->
        AnalysisSession.mutate(session, fn ->
          assert_raise ArgumentError, ~r/AnalysisMutationInProgress/, fn -> AnalysisSession.stats(session) end
          assert_raise Kinda.CallError, ~r/AnalysisMutationInProgress/, fn -> AnalysisSession.close(session) end
          raise "mutation failed"
        end)
      end

      assert %{generation: 3, cached: []} = AnalysisSession.stats(session)
      module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!(analysis_session: session)
      assert %{generation: 5, cached: []} = AnalysisSession.stats(session)
    end)
  end

  test "greedy rewriting invalidates analyses for mutable and frozen pattern sets", %{
    ctx: ctx,
    module: module,
    add: add,
    mul: mul
  } do
    AnalysisSession.with(module, fn session ->
      for {kind, generation} <- [mutable: 2, frozen: 4] do
        assert AnalysisSession.dominates?(session, add, mul)
        marker = MLIR.Attribute.string(Atom.to_string(kind), ctx: ctx)
        program = Beaver.Pattern.Native.Program.new("arith.muli", rewrite: {:attributes, marker: marker})
        set = MLIR.RewritePatternSet.create(ctx) |> MLIR.RewritePatternSet.add(program, ctx: ctx)
        set = if kind == :frozen, do: MLIR.RewritePatternSet.freeze(set), else: set

        try do
          assert {:ok, ^module} = MLIR.apply_(module, set, analysis_session: session, enable_folding: false)
          assert MLIR.equal?(mul[:marker], marker)
          assert %{generation: ^generation, cached: []} = AnalysisSession.stats(session)
        after
          if kind == :frozen, do: MLIR.RewritePatternSet.threaded_destroy(ctx, set)
        end
      end
    end)
  end

  test "checks ownership, query scope and closed sessions", %{module: module, add: add, ctx: ctx} do
    session = AnalysisSession.create(module)

    assert Task.async(fn ->
             assert_raise ArgumentError, ~r/another process/, fn -> AnalysisSession.stats(session) end
           end)
           |> Task.await()

    other = MLIR.Module.create!("module {}", ctx: ctx)

    assert_raise ArgumentError, ~r/InvalidAnalysisQuery/, fn ->
      AnalysisSession.dominates?(session, add, Operation.from_module(other))
    end

    assert :ok = AnalysisSession.close(session)
    assert :ok = AnalysisSession.close(session)
    assert_raise ArgumentError, ~r/ClosedAnalysisSession/, fn -> AnalysisSession.stats(session) end
    MLIR.Module.destroy(other)
  end

  test "DataLayout caches scoped index widths and retains scalable sizes", %{ctx: ctx} do
    module =
      "module {\n  module @narrow attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>} {}\n  module @wide attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {}\n}\n"
      |> MLIR.Module.create!(ctx: ctx)

    [narrow, wide] = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.to_list()

    AnalysisSession.with(module, fn session ->
      index = Type.index(ctx: ctx)

      assert %{size: %{minimum: 4, scalable?: false}, size_in_bits: %{minimum: 32}} =
               DataLayout.type_layout(session, index, at: narrow)

      assert %{size_in_bits: %{minimum: 64}} = DataLayout.type_layout(session, index, at: wide)

      assert %{size_in_bits: %{minimum: 128, scalable?: true}} =
               DataLayout.type_layout(session, Type.get("vector<[4]xi32>", ctx: ctx))

      AnalysisSession.mutate(session, fn ->
        MLIR.Dialect.DLTI.attach(narrow, MLIR.Dialect.DLTI.spec([{index, 64}], ctx: ctx))
      end)

      assert %{size_in_bits: %{minimum: 64}} = DataLayout.type_layout(session, index, at: narrow)

      assert_raise ArgumentError, ~r/UnsupportedLayoutQuery/, fn ->
        DataLayout.type_layout(session, Type.get(~s[!llvm.struct<"opaque", opaque>], ctx: ctx))
      end

      assert_raise ArgumentError, ~r/UnsupportedLayoutQuery/, fn ->
        DataLayout.type_layout(session, Type.get("tensor<4xi32>", ctx: ctx))
      end
    end)

    MLIR.Module.destroy(module)
  end
end
