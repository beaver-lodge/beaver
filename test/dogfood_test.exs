defmodule Beaver.MLIR.DogfoodDialect do
  @moduledoc false
  use Beaver.Slang, name: "dogfood"

  alias Beaver.MLIR.Type

  defop square(value = Type.i32()), do: [Type.i32()]
  defop inc(value = Type.i32()), do: [Type.i32()]
end

defmodule Beaver.MLIR.DogfoodNativePatterns do
  @moduledoc false
  use Beaver
  use Beaver.Pattern.Native

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Arith
  alias Beaver.MLIR.Type

  defrewrite square_to_mul(operation, rewriter, _state),
    root: "dogfood.square",
    operands: [input],
    results: [_result],
    attributes: [] do
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(operation)
    type = MLIR.Value.type(input)

    mlir ctx: ctx, ip: base do
      replacement = Arith.muli(input, input) >>> type
      MLIR.RewriterBase.replace_op(base, operation, [replacement])
    end

    :ok
  end

  defrewrite inc_to_add1(operation, rewriter, _state),
    root: "dogfood.inc",
    operands: [input],
    results: [_result],
    attributes: [] do
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(operation)
    type = MLIR.Value.type(input)

    mlir ctx: ctx, ip: base do
      one = Arith.constant(value: MLIR.Attribute.integer(type, 1)) >>> type
      replacement = Arith.addi(input, one) >>> type
      MLIR.RewriterBase.replace_op(base, operation, [replacement])
    end

    :ok
  end
end

defmodule Beaver.MLIR.DogfoodSchedule do
  @moduledoc false
  use Beaver.MLIR.Transform.Schedule.DSL

  defschedule canonicalize do
    sequence "__transform_main", [_root >>> any_op()] do
      _tile = knob("tile_size", [8, 16])
    end
  end
end

defmodule DogfoodTest.TelemetryHandler do
  @moduledoc false
  def handle_event(event, measurements, metadata, %{parent: parent}) do
    send(parent, {:dogfood_telemetry, event, measurements, metadata})
  end

  def handle_event(_event, _measurements, _metadata, _config), do: :ok
end

defmodule DogfoodTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR
  alias MLIR.CompilationCache
  alias MLIR.CompilationPlan
  alias MLIR.CompilationRuntime
  alias MLIR.Conversion.Plan
  alias MLIR.Transform.Schedule
  alias MLIR.DogfoodDialect
  alias MLIR.DogfoodNativePatterns
  alias MLIR.DogfoodSchedule

  @source """
  module {
    func.func @kernel(%arg0: i32, %arg1: i32) -> (i32, i32) {
      %0 = "dogfood.square"(%arg0) : (i32) -> i32
      %1 = "dogfood.inc"(%arg1) : (i32) -> i32
      return %0, %1 : i32, i32
    }
  }
  """

  test "dogfood: Slang -> native rewrite -> conversion plan -> schedule DSL -> compilation plan",
       %{
         ctx: ctx
       } do
    assert Beaver.Slang.load(ctx, DogfoodDialect) |> MLIR.LogicalResult.success?()

    module = MLIR.Module.create!(@source, ctx: ctx)

    # 1. #485: native matcher rewrites dogfood.square without hand-written
    #    Walker unpacking and without generating wrapper source.
    MLIR.Rewrite.apply_patterns!(module, [DogfoodNativePatterns.square_to_mul()])

    rewritten = MLIR.to_string(module)
    refute rewritten =~ "dogfood.square"
    assert rewritten =~ "arith.muli"

    # 2. #486: a scoped Conversion Plan lowers dogfood.inc, reusing the same
    #    native matcher as its conversion pattern frontend.
    plan =
      Plan.new(
        mode: :full,
        folding_mode: :after_patterns,
        build_materializations: true,
        timeout: 5_000
      )
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_legal_dialect("func")
      |> Plan.add_legal_dialect("arith")
      |> Plan.add_illegal_dialect("dogfood")
      |> Plan.add_conversion(fn type -> type end, version: "1.0")
      |> Plan.add_pattern(DogfoodNativePatterns.inc_to_add1())

    assert {:ok, ^module, _diagnostics} = Plan.run(plan, module)

    lowered = MLIR.to_string(module)
    refute lowered =~ "dogfood.inc"
    assert lowered =~ "arith.addi"
    MLIR.verify!(module)

    # The same plan declaration is reusable in a fresh MLIR context.
    ctx2 = MLIR.Context.create()
    assert Beaver.Slang.load(ctx2, DogfoodDialect) |> MLIR.LogicalResult.success?()
    module2 = MLIR.Module.create!(@source, ctx: ctx2)
    MLIR.Rewrite.apply_patterns!(module2, [DogfoodNativePatterns.square_to_mul()])
    assert {:ok, _module2, _diagnostics} = Plan.run(plan, module2)
    refute MLIR.to_string(module2) =~ "dogfood"
    MLIR.Context.destroy(ctx2)

    # 3. #488: defschedule authors upstream Transform IR (no Beaver-private
    #    schedule object) that the CompilationPlan consumes by identity.
    schedule = own(DogfoodSchedule.canonicalize(ctx: ctx))
    resolved = Schedule.resolve!(schedule, %{"tile_size" => 8})
    schedule_text = resolved.text
    assert schedule_text =~ "transform.with_named_sequence"

    # 4. #487: a cache-stable CompilationPlan carries the conversion plan and
    #    schedule identities in its deterministic telemetry metadata.
    plan_identity = inspect(Schedule.cache_identity(resolved))

    compilation_plan =
      CompilationPlan.new(
        pipeline: ["canonicalize"],
        target: %{triple: "host"},
        schema_version: "v1",
        telemetry_metadata: %{
          dogfood: true,
          conversion_plan: "dogfood.inc -> arith.addi",
          schedule_identity: plan_identity
        }
      )
      |> CompilationPlan.set_transform_schedule(schedule_text)

    parent = self()

    :telemetry.attach_many(
      "dogfood-test",
      [
        [:beaver, :mlir, :compilation, :cache, :miss],
        [:beaver, :mlir, :compilation, :cache, :hit]
      ],
      &DogfoodTest.TelemetryHandler.handle_event/4,
      %{parent: parent}
    )

    try do
      cache = start_supervised!({CompilationCache.Memory, []})

      first = CompilationRuntime.compile!(module, compilation_plan, cache: cache)
      second = CompilationRuntime.compile!(module, compilation_plan, cache: cache)

      assert first.cache == :miss
      assert second.cache == :hit
      assert first.key == second.key
      assert first.bytecode == second.bytecode
      assert second.timings.parse == 0
      assert second.timings.transform == 0

      # 5. telemetry correlates the compilation plan with the conversion and
      #    schedule identities.
      telemetry =
        Enum.map(0..50, fn _ ->
          receive do
            {:dogfood_telemetry, event, _measurements, metadata} -> {event, metadata}
          after
            0 -> nil
          end
        end)
        |> Enum.reject(&is_nil/1)

      assert telemetry != []

      cache_events =
        Enum.filter(telemetry, fn
          {[_, _, _, :cache, _], _metadata} -> true
          _ -> false
        end)

      assert Enum.any?(cache_events, fn {event, _} ->
               event == [:beaver, :mlir, :compilation, :cache, :miss]
             end)

      assert Enum.any?(cache_events, fn {event, _} ->
               event == [:beaver, :mlir, :compilation, :cache, :hit]
             end)

      miss_plan_id =
        cache_events
        |> Enum.find_value(fn
          {[:beaver, :mlir, :compilation, :cache, :miss], metadata} -> metadata[:plan_id]
          _ -> nil
        end)

      hit_plan_id =
        cache_events
        |> Enum.find_value(fn
          {[:beaver, :mlir, :compilation, :cache, :hit], metadata} -> metadata[:plan_id]
          _ -> nil
        end)

      refute is_nil(miss_plan_id)
      refute is_nil(hit_plan_id)
      assert miss_plan_id == hit_plan_id

      assert Enum.any?(telemetry, fn {_event, metadata} ->
               metadata[:dogfood] == true and
                 metadata[:conversion_plan] == "dogfood.inc -> arith.addi" and
                 metadata[:schedule_identity] == plan_identity
             end)
    after
      :telemetry.detach("dogfood-test")
    end

    MLIR.Module.destroy(module)
  end

  defp own(module) do
    on_exit(fn -> MLIR.Module.destroy(module) end)
    module
  end
end
