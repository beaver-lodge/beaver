defmodule CustomTraitTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR

  @moduletag :smoke

  test "Slang attaches custom traits with a shared identity", %{ctx: ctx} do
    assert CustomTraitSlang
           |> then(&Beaver.Slang.load(ctx, &1))
           |> MLIR.LogicalResult.success?()

    assert MLIR.Trait.has?(ctx, "custom_trait_test.valid", CustomTraitSlang.Validity)
    assert MLIR.Trait.has?(ctx, "custom_trait_test.also_valid", CustomTraitSlang.Validity)

    valid = module_with(ctx, "custom_trait_test.valid")
    assert {:ok, ^valid} = MLIR.verify(valid)
  end

  test "ordinary verifier failures become MLIR diagnostics", %{ctx: ctx} do
    Beaver.Slang.load(ctx, CustomTraitSlang)

    assert {:error, diagnostics} = verify_with(ctx, "custom_trait_test.invalid")
    rendered = MLIR.Diagnostic.format(diagnostics)
    assert rendered =~ "CustomTraitSlang.Validity"
    assert rendered =~ "invalid_operation"
  end

  test "region verifier exceptions become MLIR diagnostics", %{ctx: ctx} do
    Beaver.Slang.load(ctx, CustomTraitSlang)

    assert {:error, diagnostics} = verify_with(ctx, "custom_trait_test.invalid_regions")
    rendered = MLIR.Diagnostic.format(diagnostics)
    assert rendered =~ "CustomTraitSlang.RegionValidity"
    assert rendered =~ "invalid regions"
  end

  test "custom TypeIDs and callback ownership are isolated by context" do
    ctx1 = MLIR.Context.create()
    ctx2 = MLIR.Context.create()

    Beaver.Slang.load(ctx1, CustomTraitSlang)
    Beaver.Slang.load(ctx2, CustomTraitSlang)

    id1 = MLIR.Trait.type_id(ctx1, CustomTraitSlang.Validity)
    id2 = MLIR.Trait.type_id(ctx2, CustomTraitSlang.Validity)

    refute MLIR.CAPI.mlirTypeIDEqual(id1, id2) |> Beaver.Native.to_term()

    attachment =
      MLIR.Trait.attach_custom(
        ctx1,
        "custom_trait_test.valid",
        __MODULE__.ManualTrait,
        verify: fn _operation -> :ok end
      )

    monitor = Process.monitor(attachment.pid)
    MLIR.Context.destroy(ctx1)
    assert_receive {:DOWN, ^monitor, :process, _, reason} when reason in [:normal, :noproc], 1_000
    MLIR.Context.destroy(ctx2)
  end

  test "callback timeout fails verification with a diagnostic", %{ctx: ctx} do
    Beaver.Slang.load(ctx, CustomTraitSlang)

    MLIR.Trait.attach_custom(
      ctx,
      "custom_trait_test.valid",
      __MODULE__.SlowTrait,
      [verify: fn _operation -> Process.sleep(100) end],
      timeout: 10
    )

    assert {:error, diagnostics} = verify_with(ctx, "custom_trait_test.valid")

    assert MLIR.Diagnostic.format(diagnostics) =~
             "dynamic trait callback timed out or its owner is unavailable"
  end

  test "a stopped callback owner fails verification safely", %{ctx: ctx} do
    Beaver.Slang.load(ctx, CustomTraitSlang)

    attachment =
      MLIR.Trait.attach_custom(
        ctx,
        "custom_trait_test.valid",
        __MODULE__.StoppedTrait,
        verify: fn _operation -> :ok end
      )

    monitor = Process.monitor(attachment.pid)
    Process.exit(attachment.pid, :kill)
    assert_receive {:DOWN, ^monitor, :process, _, :killed}, 1_000

    assert {:error, diagnostics} = verify_with(ctx, "custom_trait_test.valid")

    assert MLIR.Diagnostic.format(diagnostics) =~
             "dynamic trait callback timed out or its owner is unavailable"
  end

  defp module_with(ctx, operation_name) do
    ctx |> operation_module(operation_name) |> MLIR.verify!()
  end

  defp verify_with(ctx, operation_name) do
    ctx |> operation_module(operation_name) |> MLIR.verify()
  end

  defp operation_module(ctx, operation_name) do
    operation = MLIR.Operation.builder(operation_name)

    mlir ctx: ctx do
      module do
        operation.() >>> []
      end
    end
  end
end
