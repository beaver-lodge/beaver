defmodule CompilationRuntimeTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias MLIR.CompilationCache
  alias MLIR.CompilationRuntime

  @source """
  module {
    func.func @add(%lhs: i32, %rhs: i32) -> i32 attributes {llvm.emit_c_interface} {
      %result = arith.addi %lhs, %rhs : i32
      return %result : i32
    }
  }
  """

  @lowering_pipeline "convert-func-to-llvm,convert-arith-to-llvm"

  setup do
    cache = start_supervised!({CompilationCache.Memory, []})
    %{cache: {:memory, cache}}
  end

  test "an unchanged compile skips parse and transforms on a cache hit", %{cache: cache} do
    parent = self()

    telemetry = fn event, measurements, metadata ->
      send(parent, {event, measurements, metadata})
    end

    opts = [cache: cache, pipeline: "canonicalize", telemetry: telemetry]

    first = CompilationRuntime.compile!(@source, opts)
    second = CompilationRuntime.compile!(@source, opts)

    assert first.cache == :miss
    assert second.cache == :hit
    assert first.key == second.key
    assert first.bytecode == second.bytecode
    assert is_integer(first.metadata.structural_hash)
    assert second.timings.parse == 0
    assert second.timings.transform == 0

    assert_received {[:beaver, :mlir, :compilation, :cache, :miss], %{count: 1}, _}
    assert_received {[:beaver, :mlir, :compilation, :cache, :hit], %{count: 1}, _}
  end

  test "all compatibility dimensions invalidate independently", %{cache: cache} do
    base = [
      cache: cache,
      pipeline: "canonicalize",
      target: %{triple: "host"},
      schema_version: 1,
      llvm_revision: "llvm-a"
    ]

    assert CompilationRuntime.compile!(@source, base).cache == :miss
    assert CompilationRuntime.compile!(@source, base).cache == :hit

    for changed <- [
          Keyword.put(base, :desired_emit_version, 0),
          Keyword.put(base, :pipeline, "cse"),
          Keyword.put(base, :target, %{triple: "other"}),
          Keyword.put(base, :schema_version, 2),
          Keyword.put(base, :llvm_revision, "llvm-b"),
          Keyword.put(base, :source, @source <> "\n")
        ] do
      {source, changed} = Keyword.pop(changed, :source, @source)
      assert CompilationRuntime.compile!(source, changed).cache == :miss
    end
  end

  test "a corrupt cache entry is invalidated and compilation falls back", %{cache: cache} do
    artifact = CompilationRuntime.compile!(@source, cache: cache)
    lookup_key = artifact.metadata.lookup_key
    :ok = CompilationCache.put(cache, lookup_key, %{format: 1, bytecode: "not bytecode"})

    repaired = CompilationRuntime.compile!(@source, cache: cache)

    assert repaired.cache == :miss
    assert String.starts_with?(repaired.bytecode, "ML\xEFR")
    assert CompilationRuntime.compile!(@source, cache: cache).cache == :hit
  end

  test "filesystem cache survives backend instances and supports invalidation" do
    root = Path.join(System.tmp_dir!(), "beaver-cache-#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf!(root) end)

    first_cache = CompilationCache.File.new(root)
    second_cache = CompilationCache.File.new(root)

    artifact = CompilationRuntime.compile!(@source, cache: first_cache)
    assert artifact.cache == :miss
    assert CompilationRuntime.compile!(@source, cache: second_cache).cache == :hit

    assert :ok = CompilationRuntime.invalidate(second_cache)
    assert CompilationRuntime.compile!(@source, cache: first_cache).cache == :miss
  end

  test "runtime functions require an explicit pipeline version", %{cache: cache} do
    assert_raise ArgumentError, ~r/pipeline_version/, fn ->
      CompilationRuntime.compile!(@source, cache: cache, pipeline: fn -> :ok end)
    end
  end

  test "an unavailable cache backend does not block compilation" do
    {:ok, cache} = CompilationCache.Memory.start_link()
    :ok = GenServer.stop(cache)

    artifact = CompilationRuntime.compile!(@source, cache: {:memory, cache})

    assert artifact.cache == :miss
    assert is_binary(artifact.bytecode)
  end

  @tag :smoke
  test "JIT and object emission consume the same normalized artifact", %{cache: cache} do
    artifact =
      CompilationRuntime.compile!(@source,
        cache: cache,
        pipeline: @lowering_pipeline,
        target: %{triple: "host"}
      )

    jit = CompilationRuntime.jit!(artifact)

    try do
      lhs = Beaver.Native.I32.make(20)
      rhs = Beaver.Native.I32.make(22)
      result = Beaver.Native.I32.make(0)

      MLIR.ExecutionEngine.invoke!(jit, "add", [lhs, rhs], result)
      assert Beaver.Native.to_term(result) == 42

      pointer = MLIR.ExecutionEngine.lookup(jit, "add")
      assert %Beaver.Native.OpaquePtr{} = pointer
      assert jit == MLIR.ExecutionEngine.register_symbol(jit, "beaver_test_add", pointer)
    after
      MLIR.ExecutionEngine.destroy(jit)
    end

    object_path =
      Path.join(System.tmp_dir!(), "beaver-runtime-#{System.unique_integer([:positive])}.o")

    on_exit(fn -> File.rm(object_path) end)

    # emit_object! returns the canonicalized (expanded) path; on Windows the
    # expansion also normalizes drive letter case and separators.
    assert CompilationRuntime.emit_object!(artifact, object_path) == Path.expand(object_path)
    assert File.stat!(object_path).size > 0
  end
end
