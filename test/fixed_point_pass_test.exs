defmodule FixedPointPassTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR

  test "declares deterministic fail-closed fixed-point pipelines" do
    fixed_point =
      MLIR.Transform.composite_fixed_point(
        name: "Canonical closure",
        pipeline: ["canonicalize"],
        max_iterations: 7
      )

    assert %MLIR.Transform.FixedPoint{
             name: "Canonical closure",
             pipeline: ["canonicalize"],
             max_iterations: 7,
             on_convergence_failure: :error
           } = fixed_point

    plan = MLIR.CompilationPlan.new(pipeline: [fixed_point])

    assert %{
             pipeline: [
               fixed_point: %{
                 name: "Canonical closure",
                 pipeline: [{:pipeline, "canonicalize"}],
                 max_iterations: 7,
                 on_convergence_failure: :error
               }
             ]
           } = MLIR.CompilationPlan.declaration(plan)
  end

  test "validates the structured declaration" do
    assert_raise KeyError, fn -> MLIR.Transform.composite_fixed_point([]) end

    assert_raise ArgumentError, ":pipeline must be a non-empty Composer pass list", fn ->
      MLIR.Transform.composite_fixed_point(pipeline: [])
    end

    assert_raise ArgumentError, ":max_iterations must be a positive integer", fn ->
      MLIR.Transform.composite_fixed_point(pipeline: ["canonicalize"], max_iterations: 0)
    end

    assert_raise ArgumentError,
                 ":on_convergence_failure must be :warn, :error, or :silent",
                 fn ->
                   apply(MLIR.Transform, :composite_fixed_point, [
                     [pipeline: ["canonicalize"], on_convergence_failure: :continue]
                   ])
                 end
  end

  test "fails the pass manager when a pipeline does not converge", %{ctx: ctx} do
    module = MLIR.Module.create!("module {}", ctx: ctx)

    toggle = fn operation ->
      if MLIR.Operation.discardable_attribute(operation, "test.toggle") do
        MLIR.Operation.remove_discardable_attribute(operation, "test.toggle")
      else
        MLIR.Operation.put_discardable_attribute(
          operation,
          "test.toggle",
          MLIR.Attribute.unit(ctx: MLIR.context(operation))
        )
      end

      :ok
    end

    unreachable = fn operation ->
      MLIR.Operation.put_discardable_attribute(
        operation,
        "test.after_fixed_point",
        MLIR.Attribute.unit(ctx: MLIR.context(operation))
      )

      :ok
    end

    fixed_point =
      MLIR.Transform.composite_fixed_point(
        name: "Toggle",
        pipeline: [{"test-toggle", "builtin.module", toggle}],
        max_iterations: 2
      )

    composer =
      module
      |> Beaver.Composer.append(fixed_point)
      |> Beaver.Composer.append({"test-after-fixed-point", "builtin.module", unreachable})

    if MLIR.Transform.composite_fixed_point_failure_action_supported?() do
      assert {:error, diagnostics} = Beaver.Composer.run(composer)
      assert MLIR.Diagnostic.format(diagnostics) =~ ~r/Toggle.*didn't converge.*2 iterations/s

      assert MLIR.Operation.discardable_attribute(
               MLIR.Operation.from_module(module),
               "test.after_fixed_point"
             ) == nil
    else
      assert_raise ArgumentError, ~r/update the LLVM pin/, fn ->
        Beaver.Composer.init(composer)
      end
    end
  end
end
