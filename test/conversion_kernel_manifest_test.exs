defmodule Beaver.MLIR.Conversion.Kernel.ManifestTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR.Conversion.Kernel.{Error, Manifest}

  @digest "sha256:" <> String.duplicate("a", 64)
  @artifact_digest "sha256:" <> String.duplicate("b", 64)

  defp manifest_map(overrides \\ %{}) do
    Map.merge(
      %{
        "schema_version" => 1,
        "compiler_kernel_abi_version" => 1,
        "provider" => "example-compiler",
        "compiler_revision" => "compiler-revision",
        "beaver_revision" => "beaver-revision",
        "llvm_revision" => "llvm-revision",
        "dialect_schema_digest" => @digest,
        "runtime_abi_digest" => "none",
        "patterns" => [
          %{"name" => "scalar.add", "root" => "example.add", "version" => "1"},
          %{"name" => "scalar.sub", "root" => "example.sub", "version" => "1"}
        ],
        "capabilities" => ["ir.create", "rewrite.replace"],
        "target" => %{
          "triple" => "aarch64-apple-darwin",
          "cpu" => "generic",
          "features" => []
        },
        "artifact_sha256" => @artifact_digest,
        "entrypoints" => %{
          "abi_version" => "example_conversion_abi_version",
          "populate" => "example_populate_patterns",
          "manifest" => "example_conversion_manifest"
        },
        "bootstrap" => %{
          "stage" => "stage1",
          "seed" => "cpp-bootstrap",
          "provenance" => "receipt:stage0-to-stage1"
        }
      },
      overrides
    )
  end

  test "round-trips canonical JSON with a stable digest" do
    manifest = Manifest.new!(manifest_map())
    encoded = Manifest.encode!(manifest)

    assert encoded == Manifest.encode!(Manifest.decode!(encoded))
    assert Manifest.to_map(manifest) == JSON.decode!(encoded)
    assert Manifest.digest(manifest) == Manifest.digest(Manifest.decode!(encoded))
    assert Manifest.digest(manifest) =~ ~r/^sha256:[0-9a-f]{64}$/
    assert Manifest.identity_digest(manifest) =~ ~r/^sha256:[0-9a-f]{64}$/

    other_artifact =
      Manifest.new!(manifest_map(%{"artifact_sha256" => "sha256:" <> String.duplicate("c", 64)}))

    refute Manifest.digest(manifest) == Manifest.digest(other_artifact)
    assert Manifest.identity_digest(manifest) == Manifest.identity_digest(other_artifact)
  end

  test "canonical JSON sorts nested object keys" do
    manifest = Manifest.new!(manifest_map())
    encoded = Manifest.encode!(manifest)

    assert encoded =~
             ~s/"entrypoints":{"abi_version":"example_conversion_abi_version","manifest":"example_conversion_manifest","populate":"example_populate_patterns"}/
  end

  test "rejects unknown fields and non-canonical pattern order" do
    assert %Error{code: :invalid_manifest, details: %{unknown: ["fallback"]}} =
             assert_raise(Error, fn ->
               Manifest.new!(Map.put(manifest_map(), "fallback", "beam-reference"))
             end)

    reversed = manifest_map(%{"patterns" => Enum.reverse(manifest_map()["patterns"])})

    assert %Error{code: :invalid_manifest, details: %{field: :pattern_names}} =
             assert_raise(Error, fn -> Manifest.new!(reversed) end)
  end

  test "rejects unsupported schema and compiler ABI versions" do
    assert %Error{code: :unsupported_schema} =
             assert_raise(Error, fn -> Manifest.new!(manifest_map(%{"schema_version" => 2})) end)

    assert %Error{code: :abi_mismatch} =
             assert_raise(Error, fn ->
               Manifest.new!(manifest_map(%{"compiler_kernel_abi_version" => 2}))
             end)
  end

  test "verifies exact identities and required capabilities" do
    manifest = Manifest.new!(manifest_map())

    assert ^manifest =
             Manifest.verify_compatible!(manifest,
               compiler_kernel_abi_version: 1,
               beaver_revision: "beaver-revision",
               llvm_revision: "llvm-revision",
               dialect_schema_digest: @digest,
               runtime_abi_digest: "none",
               target: manifest.target,
               capabilities: ["ir.create"]
             )

    assert %Error{code: :llvm_revision_mismatch} =
             assert_raise(Error, fn ->
               Manifest.verify_compatible!(manifest, llvm_revision: "other")
             end)

    assert %Error{code: :capability_missing, details: %{missing: ["region.take_body"]}} =
             assert_raise(Error, fn ->
               Manifest.verify_compatible!(manifest, capabilities: ["region.take_body"])
             end)
  end

  @tag :tmp_dir
  test "verifies artifact bytes and fails closed on missing or mismatched files", %{
    tmp_dir: tmp_dir
  } do
    path = Path.join(tmp_dir, "kernel.bin")
    File.write!(path, "kernel")
    digest = "sha256:" <> Base.encode16(:crypto.hash(:sha256, "kernel"), case: :lower)
    manifest = Manifest.new!(manifest_map(%{"artifact_sha256" => digest}))

    assert ^manifest = Manifest.verify_artifact!(manifest, path)

    assert %Error{code: :artifact_digest_mismatch} =
             assert_raise(Error, fn ->
               Manifest.verify_artifact!(Manifest.new!(manifest_map()), path)
             end)

    assert %Error{code: :artifact_unreadable} =
             assert_raise(Error, fn -> Manifest.verify_artifact!(manifest, path <> ".missing") end)
  end
end
