defmodule Beaver.MLIR.Transform.ScheduleTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias MLIR.Transform.{Schedule, Tuner}

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

  @conditional_schedule """
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op) {
      transform.tune.alternatives<"route"> {
        %fast = transform.tune.knob<"fast_tile"> options = [8, 16] -> !transform.any_param
        transform.yield
      }, {
        %slow = transform.tune.knob<"slow_tile"> options = [32, 64] -> !transform.any_param
        transform.yield
      }
      transform.yield
    }
  }
  """

  @constraint_schedule """
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op) {
      %tile = transform.param.constant 42 -> !transform.param<i64>
      transform.smt.constrain_params(%tile) : (!transform.param<i64>) -> () {
        ^bb0(%symbol: !smt.int):
        %zero = smt.int.constant 0
        %in_range = smt.int.cmp le %zero, %symbol
        smt.assert %in_range
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

  test "Tune choices enumerate conditionally and resolve to replayable IR", %{ctx: ctx} do
    assert {:ok, analysis} = Schedule.analyze(@conditional_schedule)
    assert Enum.map(analysis.choices, & &1.name) == ["route", "fast_tile", "slow_tile"]

    assert {:ok,
            [
              %{"route" => 0, "fast_tile" => 8},
              %{"route" => 0, "fast_tile" => 16},
              %{"route" => 1, "slow_tile" => 32},
              %{"route" => 1, "slow_tile" => 64}
            ]} = Schedule.enumerate(@conditional_schedule)

    resolved =
      Schedule.resolve!(@conditional_schedule, %{"route" => 1, "slow_tile" => 64})

    assert resolved.choices == %{"route" => 1, "slow_tile" => 64}
    assert resolved.text =~ ~s(selected_region = 1)
    assert resolved.text =~ ~s(transform.tune.knob<"slow_tile"> = 64)
    assert String.starts_with?(Schedule.serialize(resolved, :bytecode), "ML\xEFR")

    replay = Schedule.resolve!(@conditional_schedule, %{"route" => 1, "slow_tile" => 64})
    assert Schedule.cache_identity(resolved) == Schedule.cache_identity(replay)

    payload = MLIR.Module.create!(@payload, ctx: ctx)

    try do
      assert {:ok, %MLIR.Transform.Result{}} = MLIR.Transform.execute(payload, resolved)

      assert {:ok, %MLIR.Transform.Result{}} =
               MLIR.Transform.execute(payload, Schedule.serialize(resolved, :bytecode))
    after
      MLIR.Module.destroy(payload)
    end
  end

  test "knob values preserve common Elixir scalar types" do
    schedule = """
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%root: !transform.any_op) {
        %coin = transform.tune.knob<"coin"> options = [true, false] -> !transform.any_param
        %animal = transform.tune.knob<"animal"> options = ["cat", unit] -> !transform.any_param
        transform.yield
      }
    }
    """

    assert {:ok, candidates} = Schedule.enumerate(schedule)

    assert candidates == [
             %{"animal" => "cat", "coin" => true},
             %{"animal" => :unit, "coin" => true},
             %{"animal" => "cat", "coin" => false},
             %{"animal" => :unit, "coin" => false}
           ]
  end

  test "SMT constraints remain inspectable and accept an optional solver adapter" do
    assert {:ok, [constraint]} = Schedule.constraints(@constraint_schedule)
    assert constraint.ir =~ ~s("transform.smt.constrain_params")
    assert constraint.operands == 1
    assert constraint.results == 0

    solver = fn constraints, selections ->
      assert length(constraints) == 1
      assert selections == %{}
      {:ok, %{solver: :test, satisfiable: true}}
    end

    assert {:ok, resolved} = Schedule.resolve(@constraint_schedule, %{}, solver: solver)
    assert resolved.constraints == [constraint]
    assert resolved.solver_metadata == %{solver: :test, satisfiable: true}

    assert {:error, %MLIR.Transform.Error{kind: :constraint_failure}} =
             Schedule.resolve(@constraint_schedule, %{},
               solver: fn _, _ -> {:error, :unsatisfiable} end
             )
  end

  test "candidate evaluation is bounded, deterministic, cancellable, and timed out" do
    evaluator = fn resolved ->
      tile = resolved.choices["tile"]
      Process.sleep(if(tile == 8, do: 15, else: 1))
      {:ok, tile * 2, %{tile: tile}}
    end

    assert {:ok, result} =
             Tuner.search(@tune_schedule, evaluator, max_concurrency: 2, timeout: 1_000)

    assert result.max_concurrency == 2

    assert Enum.map(result.candidates, & &1.choices) == [
             %{"tile" => 8, "vectorize" => 0},
             %{"tile" => 8, "vectorize" => 1},
             %{"tile" => 16, "vectorize" => 0},
             %{"tile" => 16, "vectorize" => 1}
           ]

    assert Enum.all?(result.candidates, &(&1.status == :ok))

    assert %Tuner.Candidate{value: 32} =
             Tuner.select(result, &Enum.max_by(&1, fn candidate -> candidate.value end))

    assert {:ok, failed} =
             Tuner.search(@tune_schedule, fn _ -> {:error, :benchmark_failed} end,
               candidates: [%{"tile" => 8, "vectorize" => 0}]
             )

    assert [%Tuner.Candidate{status: :evaluation_failure, reason: :benchmark_failed}] =
             failed.candidates

    assert {:ok, timed_out} =
             Tuner.search(@tune_schedule, fn _ -> Process.sleep(50) end,
               max_concurrency: 2,
               timeout: 1
             )

    assert Enum.all?(timed_out.candidates, &(&1.status == :timeout))

    cancellation = Tuner.Cancellation.new()
    :ok = Tuner.Cancellation.cancel(cancellation)

    assert {:ok, cancelled} =
             Tuner.search(@tune_schedule, fn _ -> flunk("must not run") end,
               cancellation: cancellation
             )

    assert Enum.all?(cancelled.candidates, &(&1.status == :cancelled))
  end

  test "autotuning telemetry correlates a candidate with MLIR action tracing", %{ctx: ctx} do
    parent = self()

    telemetry = fn event, measurements, metadata ->
      send(parent, {:autotuning_telemetry, event, measurements, metadata})
    end

    choices = %{"tile" => 8, "vectorize" => 0}

    evaluator = fn _resolved, candidate_context ->
      session =
        MLIR.ActionTracing.attach(ctx,
          telemetry: telemetry,
          metadata: candidate_context.telemetry_metadata
        )

      module = MLIR.Module.create!("module {}", ctx: ctx)

      try do
        module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()
        assert MLIR.ActionTracing.drain(session) != []
        {:ok, candidate_context.index}
      after
        MLIR.ActionTracing.detach(session)
        MLIR.Module.destroy(module)
      end
    end

    assert {:ok, result} =
             Tuner.search(@tune_schedule, evaluator,
               candidates: [choices],
               max_concurrency: 1,
               telemetry: telemetry
             )

    assert [%Tuner.Candidate{status: :ok, schedule: %{digest: digest}}] = result.candidates

    assert_receive {:autotuning_telemetry, [:beaver, :mlir, :compilation, :autotuning, :start],
                    %{}, %{candidate_count: 1}}

    assert_receive {:autotuning_telemetry,
                    [:beaver, :mlir, :compilation, :autotuning, :candidate, :start], %{},
                    %{candidate_index: 0, choices: ^choices}}

    assert_receive {:autotuning_telemetry, [:beaver, :mlir, :compilation, :action, :stop],
                    %{duration: action_duration},
                    %{candidate_index: 0, choices: ^choices, transform_schedule_digest: ^digest}}

    assert is_integer(action_duration) and action_duration >= 0

    assert_receive {:autotuning_telemetry,
                    [:beaver, :mlir, :compilation, :autotuning, :candidate, :stop],
                    %{duration: candidate_duration},
                    %{candidate_index: 0, status: :ok, transform_schedule_digest: ^digest}}

    assert is_integer(candidate_duration) and candidate_duration >= 0

    assert_receive {:autotuning_telemetry, [:beaver, :mlir, :compilation, :autotuning, :stop],
                    %{candidate_count: 1}, %{status_counts: %{ok: 1}}}
  end

  test "resolved schedules participate in incremental compilation cache identity" do
    cache = start_supervised!({MLIR.CompilationCache.Memory, []})
    cache = {:memory, cache}
    resolved = Schedule.resolve!(@canonicalize_schedule, %{})

    first =
      MLIR.CompilationRuntime.compile!(@payload,
        cache: cache,
        transform_schedule: resolved
      )

    second =
      MLIR.CompilationRuntime.compile!(@payload,
        cache: cache,
        transform_schedule: resolved
      )

    assert first.cache == :miss
    assert second.cache == :hit
    assert first.metadata.transform_schedule == Schedule.cache_identity(resolved)
    assert first.metadata.transform_options == %{}

    changed =
      MLIR.CompilationRuntime.compile!(@payload,
        cache: cache,
        transform_schedule: resolved,
        transform_options: [expensive_checks: false]
      )

    assert changed.cache == :miss

    assert_raise ArgumentError, ":transform_options must be a keyword list", fn ->
      MLIR.CompilationRuntime.compile!(@payload,
        cache: cache,
        transform_schedule: resolved,
        transform_options: %{expensive_checks: false}
      )
    end
  end
end
