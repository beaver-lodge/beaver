defmodule Beaver.InstallPrebuiltLlvmTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureIO

  alias Mix.Tasks.Beaver.InstallPrebuiltLlvm

  test "toolchain metadata is valid and retains the kinda.ref compatibility pin" do
    metadata = Beaver.ToolchainMetadata.load!()

    kinda_ref =
      Beaver.ToolchainMetadata.path()
      |> Path.dirname()
      |> Path.join("kinda.ref")
      |> File.stream!()
      |> Enum.find(&(String.trim(&1) != "" and not String.starts_with?(String.trim(&1), "#")))
      |> String.trim()

    assert metadata["schema_version"] == 3
    assert metadata["kinda"]["ref"] == kinda_ref

    assert metadata["llvm"]["source_ref"] == "378916e7f7c77cce03180b3c9e4d01de3c55c0f9"
    assert map_size(metadata["llvm"]["artifacts"]) == 4
  end

  test "print-metadata emits the machine-readable toolchain contract" do
    output = capture_io(fn -> InstallPrebuiltLlvm.run(["--print-metadata"]) end)
    assert JSON.decode!(output) == Beaver.ToolchainMetadata.load!()
  end

  test "default resolution selects a checksum-pinned artifact for the required source revision" do
    output =
      capture_io(fn ->
        InstallPrebuiltLlvm.run([
          "--resolve-only",
          "--asset-os",
          "manylinux",
          "--asset-arch",
          "x86_64"
        ])
      end)

    assert output =~ "mlir_manylinux_x86_64_378916e7f7c77cce03180b3c9e4d01de3c55c0f9.zip"
  end

  @tag :tmp_dir
  test "artifact installation checks content, source identity and archive paths", %{tmp_dir: tmp} do
    ref = "378916e7f7c77cce03180b3c9e4d01de3c55c0f9"
    zip = Path.join(tmp, "artifact.zip")

    marker =
      if match?({:win32, _}, :os.type()),
        do: ~c"llvm-install/bin/llvm-config.exe",
        else: ~c"llvm-install/bin/llvm-config"

    files = [
      {marker, "binary"},
      {~c"llvm-install/include/llvm/Support/VCSRevision.h", "#define LLVM_REVISION \"#{ref}\""}
    ]

    {:ok, _} = :zip.create(String.to_charlist(zip), files)
    digest = :crypto.hash(:sha256, File.read!(zip)) |> Base.encode16(case: :lower)
    root = Beaver.LLVM.extract!(zip, Path.join(tmp, "valid"), digest)
    assert :ok = Beaver.LLVM.verify_revision!(root, ref)

    assert_raise Mix.Error, ~r/source revision mismatch/, fn ->
      Beaver.LLVM.verify_revision!(root, "different")
    end

    assert_raise Mix.Error, ~r/SHA-256 mismatch/, fn ->
      Beaver.LLVM.extract!(zip, Path.join(tmp, "bad"), String.duplicate("0", 64))
    end

    refute File.exists?(Path.join(tmp, "bad"))
    unsafe = Path.join(tmp, "unsafe.zip")
    {:ok, _} = :zip.create(String.to_charlist(unsafe), [{~c"../escape", "no"}])
    digest = :crypto.hash(:sha256, File.read!(unsafe)) |> Base.encode16(case: :lower)

    assert_raise Mix.Error, ~r/unsafe LLVM artifact path/, fn ->
      Beaver.LLVM.extract!(unsafe, Path.join(tmp, "escape"), digest)
    end
  end
end
