defmodule Beaver.MLIR.ConversionResultTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias MLIR.Conversion.{Plan, Result}

  defp plan(mode) do
    Plan.new(mode: mode)
    |> Plan.add_legal_dialect("builtin")
    |> Plan.add_illegal_op("report.drop")
    |> Plan.add_conversion_pattern("report.drop", fn op, [], rewriter ->
      rewriter |> MLIR.ConversionPatternRewriter.as_base() |> MLIR.RewriterBase.erase_op(op)
      :ok
    end)
  end

  test "analysis simulates rewrites, reports original identities and leaves IR unchanged", %{
    ctx: ctx
  } do
    MLIR.Context.allow_unregistered_dialects(ctx)

    module =
      MLIR.Module.create!(~s[module { "report.drop"() : () -> () "report.keep"() : () -> () }],
        ctx: ctx
      )

    original = MLIR.to_string(module)

    assert {:ok,
            %Result{mode: :analysis, status: :succeeded, changed?: false, rolled_back?: true} =
              report} =
             Plan.analyze(plan(:full), module)

    assert MLIR.to_string(module) == original
    assert Enum.any?(report.legalizable, &(&1["name"] == "report.drop" and &1["index"] == 1))
    refute Enum.any?(report.legalizable, &(&1["name"] == "report.keep"))
    MLIR.Module.destroy(module)
    assert Enum.all?(report.legalizable, &is_binary(&1["location"]))
  end

  test "partial conversion reports remaining original ops and committed changes", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)

    module =
      MLIR.Module.create!(~s[module { "report.drop"() : () -> () "report.keep"() : () -> () }],
        ctx: ctx
      )

    assert {:ok,
            %Result{
              status: :succeeded,
              changed?: true,
              rolled_back?: false,
              unlegalized_complete?: true
            } = report} =
             Plan.report(plan(:partial), module)

    assert [%{"name" => "report.keep", "index" => 2}] = report.unlegalized
    refute MLIR.to_string(module) =~ "report.drop"
    MLIR.Module.destroy(module)
  end

  test "failed full conversion reports rollback even after a successful speculative rewrite", %{
    ctx: ctx
  } do
    MLIR.Context.allow_unregistered_dialects(ctx)

    module =
      MLIR.Module.create!(~s[module { "report.drop"() : () -> () "report.keep"() : () -> () }],
        ctx: ctx
      )

    original = MLIR.to_string(module)

    assert {:error,
            %MLIR.Conversion.Error{
              result: %Result{status: :failed, changed?: false, rolled_back?: true}
            }} =
             Plan.report(plan(:full), module)

    assert MLIR.to_string(module) == original
    MLIR.Module.destroy(module)
  end

  test "successful and no-op full conversions have distinct mutation results", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)
    module = MLIR.Module.create!(~s[module { "report.drop"() : () -> () }], ctx: ctx)

    assert {:ok, %Result{status: :succeeded, changed?: true, rolled_back?: false}} =
             Plan.report(plan(:full), module)

    assert {:ok, %Result{status: :succeeded, changed?: false, rolled_back?: false}} =
             Plan.report(plan(:full), module)

    MLIR.Module.destroy(module)
  end
end
