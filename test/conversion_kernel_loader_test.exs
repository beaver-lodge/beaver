defmodule Beaver.MLIR.Conversion.KernelLoaderTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Kernel.{Error, Manifest}
  alias Beaver.MLIR.Conversion.Plan

  @digest "sha256:" <> String.duplicate("a", 64)
  @dummy_artifact_digest "sha256:" <> String.duplicate("b", 64)
  @fixture Path.expand("fixtures/compiler_kernel_fixture.c", __DIR__)
  @include_dir Path.expand("../native/include", __DIR__)

  setup do
    ctx = MLIR.Context.create()
    MLIR.Context.allow_unregistered_dialects(ctx)
    on_exit(fn -> MLIR.Context.destroy(ctx) end)
    %{ctx: ctx}
  end

  @tag :tmp_dir
  test "loads a content-addressed kernel and declares it without its local path", %{
    ctx: ctx,
    tmp_dir: tmp_dir
  } do
    {manifest, artifact} = build_fixture!(tmp_dir)

    plan =
      Plan.new(mode: :full)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_legal_dialect("func")
      |> Plan.add_legal_dialect("arith")
      |> Plan.add_conversion_map(["i64"], "i64")
      |> Plan.add_external_pattern_population(manifest, artifact,
        expected: expected_identities(manifest)
      )

    module =
      MLIR.Module.create!(
        ~s[module { func.func @add(%a: i64, %b: i64) -> i64 { %0 = "fixture.add"(%a, %b) : (i64, i64) -> i64 func.return %0 : i64 } }],
        ctx: ctx
      )

    assert {:ok, ^module, []} = Plan.run(plan, module)
    rendered = MLIR.to_string(module)
    refute rendered =~ "fixture.add"
    assert rendered =~ "arith.addi"

    declaration = List.last(Plan.declaration(plan).entries)
    assert declaration.kind == :add_external_pattern_population
    assert declaration.provider == "fixture"
    assert declaration.artifact_sha256 == manifest.artifact_sha256
    assert declaration.manifest_digest == Manifest.digest(manifest)
    assert declaration.identity_digest == Manifest.identity_digest(manifest)
    refute inspect(declaration) =~ artifact

    MLIR.Module.destroy(module)
  end

  @tag :tmp_dir
  test "fails closed on artifact ABI and embedded identity mismatches", %{
    ctx: ctx,
    tmp_dir: tmp_dir
  } do
    {abi_manifest, abi_artifact} = build_fixture!(Path.join(tmp_dir, "abi"), abi: 2)

    assert %Error{code: :abi_mismatch} =
             assert_raise(Error, fn -> run_kernel!(ctx, abi_manifest, abi_artifact) end)

    wrong_identity = "sha256:" <> String.duplicate("f", 64)

    {identity_manifest, identity_artifact} =
      build_fixture!(Path.join(tmp_dir, "identity"), identity: wrong_identity)

    assert %Error{code: :manifest_identity_mismatch} =
             assert_raise(Error, fn ->
               run_kernel!(ctx, identity_manifest, identity_artifact)
             end)
  end

  @tag :tmp_dir
  test "fails closed on missing symbols and pattern manifest drift", %{
    ctx: ctx,
    tmp_dir: tmp_dir
  } do
    missing_entrypoints = %{
      "abi_version" => "fixture_abi_version",
      "manifest" => "fixture_manifest",
      "populate" => "fixture_missing_populate"
    }

    {symbol_manifest, symbol_artifact} =
      build_fixture!(Path.join(tmp_dir, "symbol"), entrypoints: missing_entrypoints)

    assert %Error{code: :missing_symbol} =
             assert_raise(Error, fn -> run_kernel!(ctx, symbol_manifest, symbol_artifact) end)

    other_patterns = [%{"name" => "fixture.other", "root" => "fixture.other", "version" => "1"}]

    {pattern_manifest, pattern_artifact} =
      build_fixture!(Path.join(tmp_dir, "pattern"), patterns: other_patterns)

    assert %Error{code: :pattern_manifest_mismatch} =
             assert_raise(Error, fn ->
               run_kernel!(ctx, pattern_manifest, pattern_artifact)
             end)
  end

  test "rejects cwd-dependent artifact paths before plan materialization" do
    manifest = Manifest.new!(manifest_map())

    assert %Error{code: :invalid_artifact_path} =
             assert_raise(Error, fn ->
               Plan.new() |> Plan.add_external_pattern_population(manifest, "kernel.so")
             end)
  end

  @tag :tmp_dir
  test "typed host calls reject invalid access and roll back partial rewrites", %{
    ctx: ctx,
    tmp_dir: tmp_dir
  } do
    for behavior <- [:bad_result_index, :partial_failure] do
      {manifest, artifact} =
        build_fixture!(Path.join(tmp_dir, to_string(behavior)), behavior: behavior)

      plan = fixture_add_plan(manifest, artifact)
      module = fixture_add_module(ctx)

      assert {:error, %MLIR.Conversion.Error{diagnostics: diagnostics}} = Plan.run(plan, module)
      assert diagnostics != []

      rendered = MLIR.to_string(module)
      assert rendered =~ "fixture.add"
      refute rendered =~ "arith.addi"
      MLIR.Module.destroy(module)
    end
  end

  @tag :tmp_dir
  test "typed host attribute calls inspect and construct scalar attributes", %{
    ctx: ctx,
    tmp_dir: tmp_dir
  } do
    patterns = [%{"name" => "fixture.attr", "root" => "fixture.attr", "version" => "1"}]

    {manifest, artifact} =
      build_fixture!(tmp_dir,
        behavior: :attribute,
        patterns: patterns,
        capabilities: ["ir.attribute.v1", "pattern.register"]
      )

    plan = fixture_add_plan(manifest, artifact)

    module =
      MLIR.Module.create!(
        ~s[module { func.func @attribute() -> i64 { %0 = "fixture.attr"() {predicate = "eq"} : () -> i64 func.return %0 : i64 } }],
        ctx: ctx
      )

    assert {:ok, ^module, []} = Plan.run(plan, module)
    rendered = MLIR.to_string(module)
    assert rendered =~ "arith.constant 42"
    refute rendered =~ "fixture.attr"
    MLIR.Module.destroy(module)

    for attributes <- ["", " {predicate = 1 : i64}"] do
      invalid =
        MLIR.Module.create!(
          ~s[module { func.func @attribute() -> i64 { %0 = "fixture.attr"()#{attributes} : () -> i64 func.return %0 : i64 } }],
          ctx: ctx
        )

      assert {:error, %MLIR.Conversion.Error{diagnostics: diagnostics}} = Plan.run(plan, invalid)
      assert diagnostics != []
      assert MLIR.to_string(invalid) =~ "fixture.attr"
      refute MLIR.to_string(invalid) =~ "arith.constant"
      MLIR.Module.destroy(invalid)
    end
  end

  defp run_kernel!(ctx, manifest, artifact) do
    plan =
      Plan.new(mode: :full)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_external_pattern_population(manifest, artifact,
        expected: expected_identities(manifest)
      )

    module = MLIR.Module.create!("module {}", ctx: ctx)

    try do
      Plan.run(plan, module)
    after
      MLIR.Module.destroy(module)
    end
  end

  defp build_fixture!(directory, opts \\ []) do
    File.mkdir_p!(directory)
    draft = Manifest.new!(manifest_map(Map.new(opts)))
    identity = Keyword.get(opts, :identity, Manifest.identity_digest(draft))
    abi = Keyword.get(opts, :abi, Manifest.abi_version())
    artifact = Path.join(directory, "compiler_kernel_fixture" <> library_extension())
    llvm_include = llvm_include_dir!()

    args =
      shared_library_args() ++
        [
          "-fPIC",
          "-I#{@include_dir}",
          "-I#{llvm_include}",
          "-DFIXTURE_ABI_VERSION=#{abi}",
          "-DFIXTURE_IDENTITY=#{inspect(identity)}",
          @fixture
        ] ++ behavior_args(opts) ++ ["-o", artifact]

    case System.cmd("zig", ["cc" | args], stderr_to_stdout: true) do
      {_, 0} -> :ok
      {output, status} -> flunk("fixture compilation failed (#{status}):\n#{output}")
    end

    artifact_digest = sha256!(artifact)

    manifest =
      Manifest.new!(manifest_map(Map.put(Map.new(opts), :artifact_sha256, artifact_digest)))

    {manifest, artifact}
  end

  defp manifest_map(overrides \\ %{}) do
    overrides =
      overrides
      |> Map.drop([:abi, :behavior, :identity])
      |> Map.new(fn {key, value} -> {to_string(key), value} end)

    Map.merge(
      %{
        "schema_version" => 1,
        "compiler_kernel_abi_version" => 1,
        "provider" => "fixture",
        "compiler_revision" => "fixture-revision",
        "beaver_revision" => "beaver-test-revision",
        "llvm_revision" => MLIR.CompilationRuntime.llvm_revision(),
        "dialect_schema_digest" => @digest,
        "runtime_abi_digest" => "none",
        "patterns" => [
          %{"name" => "fixture.add", "root" => "fixture.add", "version" => "1"}
        ],
        "capabilities" => ["pattern.register"],
        "target" => %{"triple" => "host-test", "cpu" => "generic", "features" => []},
        "artifact_sha256" => @dummy_artifact_digest,
        "entrypoints" => %{
          "abi_version" => "fixture_abi_version",
          "manifest" => "fixture_manifest",
          "populate" => "fixture_populate"
        },
        "bootstrap" => %{
          "stage" => "stage1",
          "seed" => "cpp-bootstrap",
          "provenance" => "test:stage0-to-stage1"
        }
      },
      overrides
    )
  end

  defp expected_identities(manifest) do
    [
      beaver_revision: manifest.beaver_revision,
      dialect_schema_digest: manifest.dialect_schema_digest,
      runtime_abi_digest: manifest.runtime_abi_digest,
      target: manifest.target,
      capabilities: manifest.capabilities
    ]
  end

  defp llvm_include_dir! do
    llvm_config =
      System.get_env("LLVM_CONFIG_PATH") || System.find_executable("llvm-config") ||
        raise "LLVM_CONFIG_PATH or llvm-config is required for compiler-kernel fixture tests"

    case System.cmd(llvm_config, ["--includedir"], stderr_to_stdout: true) do
      {include_dir, 0} -> String.trim(include_dir)
      {output, status} -> raise "llvm-config --includedir failed (#{status}): #{output}"
    end
  end

  defp shared_library_args do
    case :os.type() do
      {:unix, :darwin} -> ["-dynamiclib"]
      _ -> ["-shared"]
    end
  end

  defp behavior_args(opts) do
    case Keyword.get(opts, :behavior) do
      nil -> []
      :bad_result_index -> ["-DFIXTURE_BAD_RESULT_INDEX"]
      :partial_failure -> ["-DFIXTURE_PARTIAL_FAILURE"]
      :attribute -> ["-DFIXTURE_ATTRIBUTE_PATTERN"]
    end
  end

  defp fixture_add_plan(manifest, artifact) do
    Plan.new(mode: :full)
    |> Plan.add_legal_dialect("builtin")
    |> Plan.add_legal_dialect("func")
    |> Plan.add_legal_dialect("arith")
    |> Plan.add_conversion_map(["i64"], "i64")
    |> Plan.add_external_pattern_population(manifest, artifact,
      expected: expected_identities(manifest)
    )
  end

  defp fixture_add_module(ctx) do
    MLIR.Module.create!(
      ~s[module { func.func @add(%a: i64, %b: i64) -> i64 { %0 = "fixture.add"(%a, %b) : (i64, i64) -> i64 func.return %0 : i64 } }],
      ctx: ctx
    )
  end

  defp library_extension do
    case :os.type() do
      {:win32, _} -> ".dll"
      {:unix, :darwin} -> ".dylib"
      _ -> ".so"
    end
  end

  defp sha256!(path) do
    "sha256:" <>
      (path |> File.read!() |> then(&:crypto.hash(:sha256, &1)) |> Base.encode16(case: :lower))
  end
end
