defmodule CompilationPlanTest do
  use ExUnit.Case, async: true

  import Beaver.MLIR.CompilationPlan
  alias Beaver.MLIR
  alias MLIR.CompilationCache
  alias MLIR.CompilationPlan
  alias MLIR.CompilationRuntime

  @source """
  module {
    func.func @add(%lhs: i32, %rhs: i32) -> i32 attributes {llvm.emit_c_interface} {
      %result = arith.addi %lhs, %rhs : i32
      return %result : i32
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

  defmodule DummyModulePass do
    def initialize(_op, _ctx), do: {:ok, nil}
    def run(op, _state), do: {:ok, op}
    def clone(state), do: state
    def destruct(_state), do: :ok
  end

  setup do
    cache = start_supervised!({CompilationCache.Memory, []})
    %{cache: {:memory, cache}}
  end

  test "deterministic declaration/identity across fresh spawned processes" do
    opts = [
      pipeline: ["canonicalize"],
      target: %{triple: "host"},
      schema_version: "v1",
      bytecode_version: :current,
      telemetry_metadata: %{app: "test"}
    ]

    plan1 = CompilationPlan.new(opts)

    task =
      Task.async(fn ->
        CompilationPlan.new(opts)
      end)

    plan2 = Task.await(task)

    assert CompilationPlan.identity(plan1) == CompilationPlan.identity(plan2)
    assert CompilationPlan.declaration(plan1) == CompilationPlan.declaration(plan2)
    assert byte_size(CompilationPlan.identity(plan1)) == 64
  end

  test "real cache hits when compiling through a plan", %{cache: cache} do
    opts = [pipeline: ["canonicalize"], target: %{triple: "host"}]
    plan = CompilationPlan.new(opts)
    equivalent = Task.async(fn -> CompilationPlan.new(opts) end) |> Task.await()

    res1 = CompilationRuntime.compile!(@source, plan, cache: cache)
    res2 = CompilationRuntime.compile!(@source, equivalent, cache: cache)

    assert res1.cache == :miss
    assert res2.cache == :hit
    assert res1.key == res2.key
    assert res1.bytecode == res2.bytecode
    assert res2.timings.parse == 0
    assert res2.timings.transform == 0
  end

  test "each invalidation dimension produces a unique identity and cache miss", %{cache: cache} do
    base =
      CompilationPlan.new(
        pipeline: ["canonicalize"],
        target: %{triple: "host"},
        schema_version: 1,
        bytecode_version: :current
      )

    res_base = CompilationRuntime.compile!(@source, base, cache: cache)
    assert res_base.cache == :miss
    assert CompilationRuntime.compile!(@source, base, cache: cache).cache == :hit

    var_pipeline =
      CompilationPlan.new(pipeline: ["cse"], target: %{triple: "host"}, schema_version: 1)

    var_nested = CompilationPlan.nested(base, "func.func", ["canonicalize"])
    var_schedule = CompilationPlan.set_transform_schedule(base, @canonicalize_schedule)

    var_schedule_options =
      CompilationPlan.set_transform_schedule(base, @canonicalize_schedule,
        expensive_checks: false
      )

    var_target = CompilationPlan.set_target(base, %{triple: "other"})
    var_schema = CompilationPlan.set_schema_version(base, 2)
    var_bytecode = CompilationPlan.set_bytecode_version(base, 0)
    var_context = CompilationPlan.set_context_options(base, allow_unregistered: true)
    var_telemetry = CompilationPlan.set_telemetry_metadata(base, compiler: :changed)
    var_versioned_pass_v1 = CompilationPlan.add_pass(base, DummyModulePass, version: 1)
    var_versioned_pass_v2 = CompilationPlan.add_pass(base, DummyModulePass, version: 2)

    plans = [
      var_pipeline,
      var_nested,
      var_schedule,
      var_schedule_options,
      var_target,
      var_schema,
      var_bytecode,
      var_context,
      var_telemetry,
      var_versioned_pass_v1,
      var_versioned_pass_v2
    ]

    ids = Enum.map([base | plans], &CompilationPlan.identity/1)
    assert length(Enum.uniq(ids)) == length(ids)

    for plan <- plans do
      res = CompilationRuntime.compile!(@source, plan, cache: cache)
      assert res.cache == :miss
    end
  end

  test "callback and module steps without version are rejected" do
    assert_raise ArgumentError, ~r/explicit :version/, fn ->
      CompilationPlan.new(pipeline: [DummyModulePass])
    end

    assert_raise ArgumentError, ~r/explicit :version/, fn ->
      CompilationPlan.new(pipeline: [fn op -> op end])
    end

    assert_raise ArgumentError, ~r/deterministic data/, fn ->
      CompilationPlan.new(target: self())
    end
  end

  test "callback declarations omit closures but retain static pass structure" do
    callback_a = fn operation -> operation end
    callback_b = fn operation -> send(self(), operation) end

    plan_a =
      CompilationPlan.new()
      |> CompilationPlan.add_pass({"callback-a", "builtin.module", callback_a}, version: 1)

    equivalent =
      CompilationPlan.new()
      |> CompilationPlan.add_pass({"callback-a", "builtin.module", callback_b}, version: 1)

    changed_root =
      CompilationPlan.new()
      |> CompilationPlan.add_pass({"callback-a", "func.func", callback_a}, version: 1)

    assert CompilationPlan.identity(plan_a) == CompilationPlan.identity(equivalent)
    refute CompilationPlan.identity(plan_a) == CompilationPlan.identity(changed_root)
    refute inspect(CompilationPlan.declaration(plan_a)) =~ "#Function"
  end

  test "execution of a versioned callback-backed Composer pass", %{cache: cache} do
    owner = self()

    run_fn = fn op ->
      send(owner, :versioned_callback_pass_ran)
      op
    end

    cb_pass = {"test_pass", "builtin.module", run_fn}

    plan = CompilationPlan.new(pipeline: [{cb_pass, version: "v1"}])

    res = CompilationRuntime.compile!(@source, plan, cache: cache)
    assert res.cache == :miss
    assert is_binary(res.bytecode)
    assert_receive :versioned_callback_pass_ran
  end

  test "plan identity is included in artifact metadata and telemetry events", %{cache: cache} do
    parent = self()

    telemetry = fn event, measurements, metadata ->
      send(parent, {event, measurements, metadata})
    end

    plan =
      CompilationPlan.new(
        pipeline: ["convert-func-to-llvm", "convert-arith-to-llvm"],
        telemetry_metadata: %{user_tag: "unit_test"}
      )

    artifact = CompilationRuntime.compile!(@source, plan, cache: cache, telemetry: telemetry)

    plan_id = CompilationPlan.identity(plan)
    assert artifact.metadata.plan_id == plan_id
    assert artifact.metadata.compilation_plan == CompilationPlan.declaration(plan)
    assert artifact.metadata.bytecode_version == :current

    assert_received {[:beaver, :mlir, :compilation, :cache, :miss], _, meta_miss}
    assert meta_miss.plan_id == plan_id
    assert meta_miss.user_tag == "unit_test"

    assert_received {[:beaver, :mlir, :compilation, :parse], _, meta_parse}
    assert meta_parse.plan_id == plan_id

    jit = CompilationRuntime.jit!(artifact, telemetry: telemetry)

    try do
      assert_received {[:beaver, :mlir, :compilation, :codegen], _, meta_codegen}
      assert meta_codegen.plan_id == plan_id
      assert meta_codegen.user_tag == "unit_test"
    after
      MLIR.ExecutionEngine.destroy(jit)
    end
  end

  test "unchanged keyword API continues to work", %{cache: cache} do
    opts = [cache: cache, pipeline: "canonicalize"]

    res1 = CompilationRuntime.compile!(@source, opts)
    res2 = CompilationRuntime.compile!(@source, opts)

    assert res1.cache == :miss
    assert res2.cache == :hit
    refute Map.has_key?(res1.metadata, :plan_id)
  end

  test "identity is recomputed from the current plan and runtime overrides cannot drift it" do
    plan = CompilationPlan.new(pipeline: "canonicalize", target: %{triple: "host"})
    mutated = %{plan | target: %{triple: "other"}}

    refute CompilationPlan.identity(plan) == CompilationPlan.identity(mutated)

    assert_raise ArgumentError, ~r/unsupported plan runtime options/, fn ->
      CompilationRuntime.compile!(@source, plan, target: %{triple: "other"})
    end
  end

  defcompiler(my_compiler,
    pipeline: ["canonicalize"],
    target: %{triple: "host"}
  )

  defcompiler block_compiler do
    CompilationPlan.new(pipeline: "cse")
  end

  test "defcompiler macro declares a plan returning function" do
    plan = my_compiler()
    assert %CompilationPlan{} = plan
    assert CompilationPlan.declaration(plan).pipeline == [{:pipeline, "canonicalize"}]
    assert CompilationPlan.declaration(block_compiler()).pipeline == [{:pipeline, "cse"}]
  end
end
