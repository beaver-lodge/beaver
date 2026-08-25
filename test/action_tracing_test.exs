defmodule ActionTracingTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR

  defmodule TelemetryHandler do
    def handle_event(_event, measurements, metadata, %{parent: parent}) do
      send(parent, {:action_stop, measurements, metadata})
    end

    def handle_event(_event, _measurements, _metadata, _config), do: :ok
  end

  @moduletag :smoke

  test "pass execution emits structured action events", %{ctx: ctx} do
    session = MLIR.ActionTracing.attach(ctx, drain_interval_ms: nil)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          func.func @f(%arg0: i32) -> i32 {
            %0 = arith.addi %arg0, %arg0 : i32
            return %0 : i32
          }
        }
        """,
        ctx: ctx
      )

    module
    |> Beaver.Composer.append("canonicalize")
    |> Beaver.Composer.run!()

    events = MLIR.ActionTracing.drain(session)

    tags = events |> Enum.map(& &1["tag"]) |> Enum.uniq()
    assert "pass-execution" in tags

    assert Enum.any?(events, &(&1["phase"] == "before"))
    assert Enum.any?(events, &(&1["phase"] == "after"))

    # Nested relationships are preserved via depth.
    assert Enum.all?(events, &is_integer(&1["depth"]))

    MLIR.ActionTracing.detach(session)
  end

  test "filtering by tag", %{ctx: ctx} do
    session = MLIR.ActionTracing.attach(ctx, tags: ["pass-execution"])

    module = MLIR.Module.create!("module {}", ctx: ctx)
    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()

    events = MLIR.ActionTracing.drain(session)

    assert events != []
    assert Enum.all?(events, &(&1["tag"] == "pass-execution"))

    MLIR.ActionTracing.detach(session)
  end

  test "filtering accepts JSON-sensitive strings before a matching tag", %{ctx: ctx} do
    session =
      MLIR.ActionTracing.attach(ctx,
        tags: ["quoted \"tag\" with \\ and\nnewline", "pass-execution"]
      )

    module = MLIR.Module.create!("module {}", ctx: ctx)
    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()

    events = MLIR.ActionTracing.drain(session)
    assert events != []
    assert Enum.all?(events, &(&1["tag"] == "pass-execution"))

    MLIR.ActionTracing.detach(session)
  end

  test "filtering by source location substring", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        ~S"""
        module {
          func.func @f() -> () {
            return
          }
        } loc("tracing.mlir":1:2)
        """,
        ctx: ctx
      )

    matching = MLIR.ActionTracing.attach(ctx, locations: ["tracing.mlir"])
    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()

    assert Enum.any?(MLIR.ActionTracing.drain(matching), &(&1["phase"] == "before"))

    non_matching = MLIR.ActionTracing.attach(ctx, locations: ["elsewhere.mlir"])
    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()

    assert MLIR.ActionTracing.drain(non_matching) == []

    MLIR.ActionTracing.detach(matching)
    MLIR.ActionTracing.detach(non_matching)
  end

  test "skip control skips the first N actions by tag", %{ctx: ctx} do
    session = MLIR.ActionTracing.attach(ctx, skip: %{"pass-execution" => 1})

    module = MLIR.Module.create!("module {}", ctx: ctx)
    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()

    # The first pass-execution action was skipped; nothing should have run.
    assert MLIR.ActionTracing.drain(session) == []

    # A second run now executes.
    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()
    events = MLIR.ActionTracing.drain(session)
    assert Enum.any?(events, &(&1["tag"] == "pass-execution" and &1["phase"] == "before"))

    MLIR.ActionTracing.detach(session)
  end

  test "control maps accept JSON-sensitive keys", %{ctx: ctx} do
    session =
      MLIR.ActionTracing.attach(ctx,
        limit: %{"a quoted \"tag\" with \\ and\nnewline" => 1, "pass-execution" => 1}
      )

    module = MLIR.Module.create!("module {}", ctx: ctx)
    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()
    assert MLIR.ActionTracing.drain(session) != []

    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()
    assert MLIR.ActionTracing.drain(session) == []

    MLIR.ActionTracing.detach(session)
  end

  test "destroying the context detaches the session" do
    ctx = MLIR.Context.create()
    session = MLIR.ActionTracing.attach(ctx)
    monitor = Process.monitor(session.pid)

    MLIR.Context.destroy(ctx)

    assert_receive {:DOWN, ^monitor, :process, _, :normal}, 1_000
  end

  test "telemetry events are emitted with duration", %{ctx: ctx} do
    parent = self()

    :telemetry.attach(
      "action-tracing-test",
      [:beaver, :mlir, :compilation, :action, :stop],
      &TelemetryHandler.handle_event/4,
      %{parent: parent}
    )

    session = MLIR.ActionTracing.attach(ctx)
    module = MLIR.Module.create!("module {}", ctx: ctx)
    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()
    MLIR.ActionTracing.drain(session)

    assert_receive {:action_stop, %{duration: duration}, %{"tag" => "pass-execution"}}
    assert is_integer(duration) and duration >= 0

    :telemetry.detach("action-tracing-test")
    MLIR.ActionTracing.detach(session)
  end

  test "drain_interval_ms emits telemetry automatically", %{ctx: ctx} do
    parent = self()

    :telemetry.attach(
      "action-tracing-interval",
      [:beaver, :mlir, :compilation, :action, :stop],
      &TelemetryHandler.handle_event/4,
      %{parent: parent}
    )

    session = MLIR.ActionTracing.attach(ctx, drain_interval_ms: 50)
    module = MLIR.Module.create!("module {}", ctx: ctx)
    module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()

    assert_receive {:action_stop, _, _}, 2_000

    :telemetry.detach("action-tracing-interval")
    MLIR.ActionTracing.detach(session)
  end
end
