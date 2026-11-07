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

    assert metadata["schema_version"] == 2
    assert metadata["kinda"]["ref"] == kinda_ref

    assert metadata["llvm"]["default_revisions"] == %{
             "macos_arm64" => "20261106+b5bbdc97c",
             "manylinux_aarch64" => "20261106+b5bbdc97c",
             "manylinux_x86_64" => "20261106+b5bbdc97c",
             "windows_amd64" => "20261106+b5bbdc97c"
           }
  end

  test "print-metadata emits the machine-readable toolchain contract" do
    output = capture_io(fn -> InstallPrebuiltLlvm.run(["--print-metadata"]) end)
    assert JSON.decode!(output) == Beaver.ToolchainMetadata.load!()
  end

  test "default resolution delegates to mlir with Beaver's selected revision" do
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

    assert output =~ "mlir_manylinux_x86_64_20261106+b5bbdc97c.tar.gz"
  end
end
