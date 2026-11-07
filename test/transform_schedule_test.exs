defmodule Beaver.MLIR.Transform.ScheduleTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias MLIR.Transform.Schedule

  @payload """
  module {
    func.func @fold_me() -> i32 {
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %sum = arith.addi %one, %zero : i32
      return %sum : i32
    }
  }
  """

  @canonicalize_schedule """
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op) {
      transform.apply_patterns to %root {
        transform.apply_patterns.canonicalization
      } : !transform.any_op
      transform.yield
    }
  }
  """

  @tune_schedule """
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op) {
      %tile = transform.tune.knob<"tile"> options = [8, 16] -> !transform.any_param
      transform.tune.alternatives<"vectorize"> {
        transform.yield
      }, {
        transform.yield
      }
      transform.yield
    }
  }
  """

  setup do
    context = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(context) end)
    %{ctx: context}
  end

  test "named sequence execution accepts operation, module, text, and bytecode", %{ctx: ctx} do
    schedule_module = MLIR.Module.create!(@canonicalize_schedule, ctx: ctx)
    {:ok, sequence_operation} = Schedule.find_sequence(schedule_module, "__transform_main")
    bytecode = MLIR.Bytecode.write!(schedule_module)

    try do
      for schedule <- [@canonicalize_schedule, bytecode, schedule_module, sequence_operation] do
        payload = MLIR.Module.create!(@payload, ctx: ctx)

        try do
          assert {:ok, %MLIR.Transform.Result{diagnostics: []}} =
                   MLIR.Transform.apply_named_sequence(payload, schedule,
                     expensive_checks: false,
                     enforce_single_top_level_transform_op: false
                   )

          rendered = MLIR.to_string(payload)
          refute rendered =~ "arith.addi"
          refute rendered =~ "arith.constant 0"
        after
          MLIR.Module.destroy(payload)
        end
      end
    after
      MLIR.Module.destroy(schedule_module)
    end
  end

  test "prepared schedules reuse parsed IR with fresh execution state", %{ctx: ctx} do
    assert {:ok, prepared} = MLIR.Transform.prepare(@canonicalize_schedule, ctx)

    try do
      for _ <- 1..3 do
        payload = MLIR.Module.create!(@payload, ctx: ctx)

        MLIR.AnalysisSession.with(payload, fn session ->
          assert {:ok, %MLIR.Transform.Result{}} =
                   MLIR.Transform.execute(payload, prepared, analysis_session: session)

          refute MLIR.to_string(payload) =~ "arith.addi"
          assert %{generation: 2, cached: []} = MLIR.AnalysisSession.stats(session)
        end)

        MLIR.Module.destroy(payload)
      end

      other = MLIR.Context.create()

      try do
        payload = MLIR.Module.create!(@payload, ctx: other)

        assert {:error, %MLIR.Transform.Error{kind: :invalid_schedule}} =
                 MLIR.Transform.execute(payload, prepared)

        MLIR.Module.destroy(payload)
      after
        MLIR.Context.destroy(other)
      end
    after
      MLIR.Transform.close(prepared)
    end

    assert :ok = MLIR.Transform.close(prepared)
    payload = MLIR.Module.create!(@payload, ctx: ctx)

    assert {:error, %MLIR.Transform.Error{reason: reason}} =
             MLIR.Transform.execute(payload, prepared)

    assert reason =~ "ClosedPreparedTransform"
    MLIR.Module.destroy(payload)
  end

  test "transform handle and parameter type helpers are context-aware", %{ctx: ctx} do
    assert MLIR.to_string(MLIR.Transform.any_op_type(ctx: ctx)) == "!transform.any_op"
    assert MLIR.to_string(MLIR.Transform.any_value_type(ctx: ctx)) == "!transform.any_value"
    assert MLIR.to_string(MLIR.Transform.any_param_type(ctx: ctx)) == "!transform.any_param"

    assert MLIR.to_string(MLIR.Transform.operation_type("func.func", ctx: ctx)) ==
             "!transform.op<\"func.func\">"

    assert MLIR.to_string(MLIR.Transform.param_type(MLIR.Type.i64(ctx: ctx))) ==
             "!transform.param<i64>"
  end

  test "execution errors keep portable diagnostics and distinct failure classes", %{ctx: ctx} do
    payload = MLIR.Module.create!(@payload, ctx: ctx)

    try do
      assert {:error, %MLIR.Transform.Error{kind: :invalid_schedule}} =
               MLIR.Transform.execute(payload, "not mlir")

      silenceable = """
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%root: !transform.any_op) {
          %func = transform.cast %root : !transform.any_op to !transform.op<"func.func">
          transform.yield
        }
      }
      """

      assert {:error,
              %MLIR.Transform.Error{
                kind: :silenceable_failure,
                diagnostics: [{:error, location, _, _}]
              }} = MLIR.Transform.execute(payload, silenceable)

      assert is_binary(location)

      assert {:error, %MLIR.Transform.Error{kind: :definite_failure}} =
               MLIR.Transform.execute(payload, @tune_schedule)
    after
      MLIR.Module.destroy(payload)
    end
  end

  test "selected Tune IR replays from text and bytecode", %{ctx: ctx} do
    selected = """
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%root: !transform.any_op) {
        %tile = transform.tune.knob<"tile"> = 16 from options = [8, 16] -> !transform.any_param
        transform.tune.alternatives<"route"> selected_region = 1 {
          transform.yield
        }, {
          transform.yield
        }
        transform.yield
      }
    }
    """

    schedule = MLIR.Module.create!(selected, ctx: ctx)
    payload = MLIR.Module.create!(@payload, ctx: ctx)

    try do
      assert {:ok, %MLIR.Transform.Result{}} = MLIR.Transform.execute(payload, {:text, selected})
      bytecode = MLIR.Bytecode.write!(schedule)

      assert {:ok, %MLIR.Transform.Result{}} =
               MLIR.Transform.execute(payload, {:bytecode, bytecode})
    after
      MLIR.Module.destroy(payload)
      MLIR.Module.destroy(schedule)
    end
  end
end
