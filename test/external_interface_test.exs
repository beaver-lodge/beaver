defmodule ExternalInterfaceTest do
  use Beaver.Case, async: true

  import ExUnit.CaptureLog

  alias Beaver.MLIR

  @moduletag :smoke

  test "Slang attaches all four external interface models", %{ctx: ctx} do
    interfaces = ExternalInterfaceSlang.__slang_interfaces__()

    assert Enum.map(interfaces, &elem(&1, 0)) ==
             ["pure", "write", "transform", "forward", "mark", "fail", "patterns"]

    assert Keyword.keys(interfaces |> hd() |> elem(1)) ==
             [:memory_effects, :conditionally_speculatable]

    assert ExternalInterfaceSlang
           |> then(&Beaver.Slang.load(ctx, &1))
           |> MLIR.LogicalResult.success?()

    assert implements?(ctx, "external_interface_test.pure", :mlirMemoryEffectsOpInterfaceTypeID)

    assert implements?(
             ctx,
             "external_interface_test.pure",
             :mlirConditionallySpeculatableOpInterfaceTypeID
           )

    assert implements?(
             ctx,
             "external_interface_test.transform",
             :mlirTransformOpInterfaceTypeID
           )

    assert implements?(
             ctx,
             "external_interface_test.transform",
             :mlirPatternDescriptorOpInterfaceTypeID
           )
  end

  test "memory effects make pure operations removable and preserve writes", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExternalInterfaceSlang)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          "external_interface_test.pure"() : () -> ()
          "external_interface_test.write"() : () -> ()
        }
        """,
        ctx: ctx
      )
      |> MLIR.verify!()

    module
    |> Beaver.Composer.append(MLIR.Transform.canonicalize())
    |> Beaver.Composer.run!()

    rendered = MLIR.to_string(module, generic: true)
    refute rendered =~ "external_interface_test.pure"
    assert rendered =~ "external_interface_test.write"
  end

  test "speculatability callbacks are serviced by their attachment process", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExternalInterfaceSlang)

    module =
      MLIR.Module.create!(
        ~S[module { "external_interface_test.pure"() : () -> () }],
        ctx: ctx
      )

    operation =
      module
      |> MLIR.Module.body()
      |> Beaver.Walker.operations()
      |> Enum.at(0)

    assert MLIR.ConditionallySpeculatable.query(operation) == :speculatable
  end

  test "destroying a context releases its attachment process" do
    ctx = MLIR.Context.create()

    attachment =
      MLIR.MemoryEffects.attach(ctx, "builtin.module", fn _operation -> :pure end)

    monitor = Process.monitor(attachment.pid)
    assert Process.alive?(attachment.pid)
    MLIR.Context.destroy(ctx)
    assert_receive {:DOWN, ^monitor, :process, _, :normal}, 1_000
  end

  test "the native model outlives an attachment process that exits abnormally", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExternalInterfaceSlang)

    module =
      MLIR.Module.create!(
        ~S[module { "external_interface_test.detached"() : () -> () }],
        ctx: ctx
      )

    attachment =
      MLIR.MemoryEffects.attach(ctx, "external_interface_test.detached", fn _operation ->
        :pure
      end)

    monitor = Process.monitor(attachment.pid)
    Process.exit(attachment.pid, :kill)
    assert_receive {:DOWN, ^monitor, :process, _, :killed}, 1_000

    module
    |> Beaver.Composer.append(MLIR.Transform.canonicalize())
    |> Beaver.Composer.run!()

    assert MLIR.to_string(module, generic: true) =~ "external_interface_test.detached"
  end

  test "Elixir transform operations map results and inject rewrite patterns", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExternalInterfaceSlang)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          func.func @payload() {
            %0 = arith.constant 1 : i32
            return
          }
          module attributes {transform.with_named_sequence} {
            transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
              %0 = "external_interface_test.forward"(%arg0) : (!transform.any_op) -> !transform.any_op
              "external_interface_test.mark"(%0) : (!transform.any_op) -> ()
              transform.apply_patterns to %0 {
                "external_interface_test.patterns"() : () -> ()
              } : !transform.any_op
              transform.yield
            }
          }
        }
        """,
        ctx: ctx
      )

    module = MLIR.verify!(module)

    module
    |> Beaver.Composer.append("transform-interpreter")
    |> Beaver.Composer.run!()

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "external_interface_test.marked"
    assert rendered =~ "external_interface_test.pattern_applied"
  end

  test "transform callback exceptions become diagnostics and definite failures", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExternalInterfaceSlang)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          func.func @payload() {
            return
          }
          module attributes {transform.with_named_sequence} {
            transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
              "external_interface_test.fail"(%arg0) : (!transform.any_op) -> ()
              transform.yield
            }
          }
        }
        """,
        ctx: ctx
      )

    log =
      capture_log(fn ->
        error =
          assert_raise ArgumentError, fn ->
            module
            |> Beaver.Composer.append("transform-interpreter")
            |> Beaver.Composer.run!()
          end

        assert Exception.message(error) =~ "external interface callback raised"
        assert Exception.message(error) =~ "expected callback failure"
      end)

    assert log =~ "expected callback failure"
  end

  defp implements?(ctx, operation_name, type_id_function) do
    type_id = apply(MLIR.CAPI, type_id_function, [])
    MLIR.Context.implements_interface?(ctx, operation_name, type_id)
  end
end
