defmodule Beaver.MLIR.Transform.Schedule.DSLTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.{SMT, Transform}
  alias Beaver.MLIR.Transform.{Schedule, Tuner}
  alias Beaver.MLIR.Transform.Schedule.DSL

  defmodule GuideSchedule do
    use DSL

    defschedule tiling_and_vectorization do
      sequence "__transform_main", [root >>> any_op()] do
        tile = knob("tile_size", [8, 16, 32], type: param(MLIR.Type.i64()))

        operation_names =
          MLIR.Attribute.array(
            [MLIR.Attribute.string("linalg.matmul")],
            ctx: Beaver.Env.context()
          )

        matmul =
          Transform.structured_match(target: root, ops: operation_names) >>> any_op()

        dynamic_tile =
          MLIR.Attribute.dense_array([-1], Beaver.Native.I64, ctx: Beaver.Env.context())

        scalable_tile =
          MLIR.Attribute.dense_array([false], Beaver.Native.Bool, ctx: Beaver.Env.context())

        [tiled, _loop] =
          Transform.structured_tile_using_for(
            target: matmul,
            dynamic_sizes: [tile],
            static_sizes: dynamic_tile,
            scalable_sizes: scalable_tile
          ) >>> [any_op(), any_op()]

        alternatives "vectorize" do
          branch do
            isolated = MLIR.Attribute.unit(ctx: Beaver.Env.context())

            function =
              Transform.get_parent_op(target: tiled, isolated_from_above: isolated) >>> any_op()

            _vectorized =
              Transform.structured_vectorize_children_and_apply_patterns(target: function) >>>
                any_op()
          end

          branch do
            :ok
          end
        end
      end
    end
  end

  defmodule ConditionalSchedule do
    use DSL

    defschedule route_schedule do
      sequence "__transform_main", [_root >>> any_op()] do
        alternatives "route" do
          branch do
            _fast = knob("fast_tile", [8, 16])
          end

          branch do
            _slow = knob("slow_tile", [32, 64])
          end
        end
      end
    end
  end

  defmodule ScalarSchedule do
    use DSL

    defschedule scalar_schedule do
      sequence "__transform_main", [_root >>> any_op()] do
        _integer = knob("integer", [8, 16])
        _float = knob("float", [0.5, 1.5])
        _boolean = knob("boolean", [true, false])
        _string = knob("string", ["fast", "slow"])
        _unit = knob("unit", [:unit])

        _attribute =
          knob("attribute", [
            fn context -> MLIR.Attribute.integer(MLIR.Type.i32(ctx: context), 42) end
          ])
      end
    end
  end

  defmodule CustomNamedSchedule do
    use DSL

    defschedule custom_schedule do
      sequence "custom_transform", [_root >>> operation("builtin.module")] do
        _tile = knob("tile", [4, 8])
      end
    end
  end

  defmodule ConstraintSchedule do
    use DSL

    defschedule constrained_schedule do
      sequence "__transform_main", [_root >>> any_op()] do
        tile =
          Transform.param_constant(
            value: MLIR.Attribute.integer(MLIR.Type.i64(ctx: Beaver.Env.context()), 42)
          ) >>> param(MLIR.Type.i64())

        Transform.smt_constrain_params params: [tile] do
          region do
            block _(_symbol >>> ~t{!smt.int}) do
              SMT.yield() >>> []
            end
          end
        end >>> []
      end
    end
  end

  defmodule CseSchedule do
    use DSL

    defschedule cse_schedule do
      sequence "__transform_main", [root >>> any_op()] do
        apply_cse = MLIR.Attribute.unit(ctx: Beaver.Env.context())

        Transform.apply_patterns target: root, apply_cse: apply_cse do
          region do
            block do
              Transform.apply_patterns_canonicalization()
            end
          end
        end >>> []
      end
    end

    defschedule plain_patterns_schedule do
      sequence "__transform_main", [root >>> any_op()] do
        Transform.apply_patterns target: root do
          region do
            block do
              Transform.apply_patterns_canonicalization()
            end
          end
        end >>> []
      end
    end
  end

  defmodule InvalidDeclarations do
    use DSL

    defschedule empty_knob do
      sequence do
        _value = knob("empty", [])
      end
    end

    defschedule unsupported_scalar do
      sequence do
        _value = knob("pid", [self()])
      end
    end

    defschedule no_sequence do
      :ok
    end

    defschedule verifier_failure do
      sequence do
        _invalid = Transform.get_parent_op(target: root) >>> param(MLIR.Type.i64())
      end
    end
  end

  @payload """
  module {
    func.func @matmul_op(%A: tensor<16x16xf32>, %B: tensor<16x16xf32>, %C: tensor<16x16xf32>) -> tensor<16x16xf32> {
      %0 = linalg.matmul ins(%A, %B : tensor<16x16xf32>, tensor<16x16xf32>) outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
      return %0 : tensor<16x16xf32>
    }
  }
  """

  @cse_payload """
  module {
    func.func @duplicated(%a: i32, %b: i32) -> i32 {
      %0 = arith.addi %a, %b : i32
      %1 = arith.addi %a, %b : i32
      %2 = arith.addi %0, %1 : i32
      return %2 : i32
    }
  }
  """

  setup do
    context = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(context) end)
    %{ctx: context}
  end

  test "authors the guide search space as verified upstream Transform IR", %{ctx: ctx} do
    schedule = own(GuideSchedule.tiling_and_vectorization(ctx: ctx))

    assert {:ok, analysis} = Schedule.analyze(schedule)
    assert Enum.map(analysis.choices, & &1.name) == ["tile_size", "vectorize"]

    assert {:ok, candidates} = Schedule.enumerate(schedule)

    assert candidates == [
             %{"tile_size" => 8, "vectorize" => 0},
             %{"tile_size" => 8, "vectorize" => 1},
             %{"tile_size" => 16, "vectorize" => 0},
             %{"tile_size" => 16, "vectorize" => 1},
             %{"tile_size" => 32, "vectorize" => 0},
             %{"tile_size" => 32, "vectorize" => 1}
           ]

    resolved = Schedule.resolve!(schedule, %{"tile_size" => 16, "vectorize" => 1})
    assert resolved.choices == %{"tile_size" => 16, "vectorize" => 1}
  end

  test "generated IR is deterministic and round-trips through text and bytecode", %{ctx: ctx} do
    first = own(ConditionalSchedule.route_schedule(ctx: ctx))
    second = own(ConditionalSchedule.route_schedule(ctx: ctx))

    assert {:ok, _} = MLIR.verify(first)
    assert MLIR.to_string(first) == MLIR.to_string(second)
    assert MLIR.Bytecode.write!(first) == MLIR.Bytecode.write!(second)

    text = MLIR.to_string(first)
    bytecode = MLIR.Bytecode.write!(first)
    assert text =~ "transform.with_named_sequence"
    assert text =~ ~s(transform.tune.alternatives<"route">)
    assert String.starts_with?(bytecode, "ML\xEFR")

    text_round_trip = own(MLIR.Module.create!(text, ctx: ctx))
    bytecode_round_trip = own(MLIR.Module.create!(bytecode, ctx: ctx))

    for round_trip <- [text_round_trip, bytecode_round_trip] do
      assert {:ok, candidates} = Schedule.enumerate(round_trip)
      assert length(candidates) == 4
    end
  end

  test "choices nested in alternatives stay conditional", %{ctx: ctx} do
    schedule = own(ConditionalSchedule.route_schedule(ctx: ctx))

    assert {:ok,
            [
              %{"route" => 0, "fast_tile" => 8},
              %{"route" => 0, "fast_tile" => 16},
              %{"route" => 1, "slow_tile" => 32},
              %{"route" => 1, "slow_tile" => 64}
            ]} = Schedule.enumerate(schedule)

    resolved = Schedule.resolve!(schedule, %{"route" => 1, "slow_tile" => 64})
    replay = Schedule.resolve!(schedule, %{"route" => 1, "slow_tile" => 64})

    assert resolved.choices == %{"route" => 1, "slow_tile" => 64}
    assert Schedule.cache_identity(resolved) == Schedule.cache_identity(replay)

    payload = own(MLIR.Module.create!(@payload, ctx: ctx))
    assert {:ok, %MLIR.Transform.Result{}} = MLIR.Transform.execute(payload, resolved)

    assert %Tuner.Result{
             candidates: [
               %Tuner.Candidate{status: :ok, choices: %{"route" => 1, "slow_tile" => 64}}
             ]
           } =
             Tuner.search!(schedule, fn _resolved, candidate -> {:ok, candidate.choices} end,
               candidates: [%{"route" => 1, "slow_tile" => 64}],
               max_concurrency: 1
             )
  end

  test "knobs preserve common Elixir scalar and MLIR attribute values", %{ctx: ctx} do
    schedule = own(ScalarSchedule.scalar_schedule(ctx: ctx))
    assert {:ok, analysis} = Schedule.analyze(schedule)

    assert Enum.map(analysis.choices, & &1.name) == [
             "integer",
             "float",
             "boolean",
             "string",
             "unit",
             "attribute"
           ]

    assert {:ok, [candidate | _]} = Schedule.enumerate(schedule)
    assert candidate["integer"] == 8
    assert candidate["float"] == 0.5
    assert candidate["boolean"] == true
    assert candidate["string"] == "fast"
    assert candidate["unit"] == :unit
    assert candidate["attribute"] == 42
  end

  test "generic SSA constraint regions remain inspectable", %{ctx: ctx} do
    schedule = own(ConstraintSchedule.constrained_schedule(ctx: ctx))
    assert {:ok, [constraint]} = Schedule.constraints(schedule)
    assert constraint.ir =~ "transform.smt.constrain_params"
    assert constraint.ir =~ "smt.yield"

    solver = fn constraints, selections ->
      assert length(constraints) == 1
      assert selections == %{}
      {:ok, %{solver: :test, satisfiable: true}}
    end

    assert {:ok, resolved} = Schedule.resolve(schedule, %{}, solver: solver)
    assert resolved.solver_metadata == %{solver: :test, satisfiable: true}
  end

  test "custom named sequences and typed handles are selectable", %{ctx: ctx} do
    schedule = own(CustomNamedSchedule.custom_schedule(ctx: ctx))
    assert {:ok, analysis} = Schedule.analyze(schedule, sequence: "custom_transform")
    assert Enum.map(analysis.choices, & &1.name) == ["tile"]
    assert MLIR.to_string(schedule) =~ ~s(!transform.op<"builtin.module">)
  end

  test "generated operations retain declaration source locations", %{ctx: ctx} do
    schedule = own(ConditionalSchedule.route_schedule(ctx: ctx))

    assert schedule
           |> MLIR.Module.body()
           |> Beaver.Walker.operations()
           |> Enum.any?(fn operation ->
             operation |> MLIR.location() |> MLIR.to_string() =~ "transform_schedule_dsl_test.exs"
           end)

    assert_raise ArgumentError, ~r/verification failed.*transform_schedule_dsl_test\.exs/s, fn ->
      InvalidDeclarations.verifier_failure(ctx: ctx)
    end
  end

  test "invalid declarations fail without taking context ownership", %{ctx: ctx} do
    assert_raise ArgumentError, ~r/requires a caller-owned/, fn ->
      InvalidDeclarations.empty_knob()
    end

    assert_raise ArgumentError, ~r/unsupported schedule options/, fn ->
      InvalidDeclarations.empty_knob(ctx: ctx, unknown: true)
    end

    assert_raise ArgumentError, ~r/knob options cannot be empty/, fn ->
      InvalidDeclarations.empty_knob(ctx: ctx)
    end

    assert_raise ArgumentError, ~r/unsupported scalar or attribute/, fn ->
      InvalidDeclarations.unsupported_scalar(ctx: ctx)
    end

    assert_raise ArgumentError, ~r/must declare at least one sequence/, fn ->
      InvalidDeclarations.no_sequence(ctx: ctx)
    end

    # Every failed builder destroyed only its temporary module; the caller's
    # context remains usable.
    assert %MLIR.Module{} = own(ConditionalSchedule.route_schedule(ctx: ctx))
  end

  test "resolved DSL schedules work unchanged with CompilationRuntime", %{ctx: ctx} do
    cache = start_supervised!({MLIR.CompilationCache.Memory, []})
    schedule = own(ConditionalSchedule.route_schedule(ctx: ctx))
    resolved = Schedule.resolve!(schedule, %{"route" => 1, "slow_tile" => 64})

    first =
      MLIR.CompilationRuntime.compile!(@payload,
        cache: {:memory, cache},
        transform_schedule: resolved
      )

    second =
      MLIR.CompilationRuntime.compile!(@payload,
        cache: {:memory, cache},
        transform_schedule: resolved
      )

    assert first.cache == :miss
    assert second.cache == :hit
    assert first.metadata.transform_schedule == Schedule.cache_identity(resolved)
  end

  defp count_arith_addi(module) do
    module
    |> MLIR.Module.body()
    |> Beaver.Walker.operations()
    |> Enum.map(&MLIR.to_string/1)
    |> Enum.join()
    |> then(&(length(String.split(&1, "arith.addi")) - 1))
  end

  test "apply_patterns with apply_cse removes duplicated subexpressions", %{ctx: ctx} do
    schedule = own(CseSchedule.cse_schedule(ctx: ctx))
    assert {:ok, resolved} = Schedule.resolve(schedule, %{})

    payload = own(MLIR.Module.create!(@cse_payload, ctx: ctx))
    assert count_arith_addi(payload) == 3

    assert {:ok, result} = MLIR.Transform.execute(payload, resolved)

    assert count_arith_addi(result.payload) == 2

    # The folded value is reused: the remaining addi feeds itself.
    assert MLIR.to_string(result.payload) =~ "arith.addi %0, %0"
  end

  test "apply_patterns without apply_cse keeps duplicated subexpressions", %{ctx: ctx} do
    schedule = own(CseSchedule.plain_patterns_schedule(ctx: ctx))

    assert {:ok, resolved} = Schedule.resolve(schedule, %{})

    payload = own(MLIR.Module.create!(@cse_payload, ctx: ctx))

    assert {:ok, result} = MLIR.Transform.execute(payload, resolved)

    assert count_arith_addi(result.payload) == 3
  end

  test "apply_cse schedules round-trip through text and bytecode", %{ctx: ctx} do
    schedule = own(CseSchedule.cse_schedule(ctx: ctx))

    text = MLIR.to_string(schedule)
    bytecode = MLIR.Bytecode.write!(schedule)
    assert text =~ ~s(transform.apply_patterns)
    assert text =~ "{apply_cse}"

    for round_trip <- [
          MLIR.Module.create!(text, ctx: ctx),
          MLIR.Module.create!(bytecode, ctx: ctx)
        ] do
      round_trip = own(round_trip)
      assert {:ok, resolved} = Schedule.resolve(round_trip, %{})

      payload = own(MLIR.Module.create!(@cse_payload, ctx: ctx))
      assert {:ok, result} = MLIR.Transform.execute(payload, resolved)

      assert count_arith_addi(result.payload) == 2
    end
  end

  defp own(module) do
    on_exit(fn -> MLIR.Module.destroy(module) end)
    module
  end
end
