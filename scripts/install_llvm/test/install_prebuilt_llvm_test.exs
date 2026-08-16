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

    assert metadata["schema_version"] == 1
    assert metadata["kinda"]["ref"] == kinda_ref
    assert metadata["llvm"]["default_revision"] =~ ~r/^\d{8}\+[0-9a-f]+$/
  end

  test "print-metadata emits the machine-readable toolchain contract" do
    output = capture_io(fn -> InstallPrebuiltLlvm.run(["--print-metadata"]) end)
    assert JSON.decode!(output) == Beaver.ToolchainMetadata.load!()
  end

  setup do
    Mix.Task.reenable("beaver.install_prebuilt_llvm")
    :ok
  end

  test "resolve-only prints the default eudsl asset" do
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

    assert output =~ "LLVM_PREBUILT_ASSET_NAME=mlir_manylinux_x86_64_20260815+8024076c7.tar.gz"
    assert output =~ "LLVM_EUDSL_ASSET_REVISION=20260815+8024076c7"
    assert output =~ "https://github.com/llvm/eudsl/releases/download/llvm/"
  end

  test "asset-url overrides GitHub asset resolution" do
    output =
      capture_io(fn ->
        InstallPrebuiltLlvm.run([
          "--resolve-only",
          "--asset-url",
          "https://example.com/llvm.tar.gz"
        ])
      end)

    assert output =~ "LLVM_PREBUILT_ASSET_NAME=llvm.tar.gz"
    assert output =~ "LLVM_PREBUILT_URL=https://example.com/llvm.tar.gz"
  end

  test "empty asset-name falls back to the default revision" do
    output =
      capture_io(fn ->
        InstallPrebuiltLlvm.run([
          "--resolve-only",
          "--asset-os",
          "manylinux",
          "--asset-arch",
          "x86_64",
          "--asset-name",
          ""
        ])
      end)

    assert output =~ "LLVM_PREBUILT_ASSET_NAME=mlir_manylinux_x86_64_20260815+8024076c7.tar.gz"
  end

  test "triton suffix maps Beaver platforms to Triton archive names" do
    assert InstallPrebuiltLlvm.triton_suffix("manylinux", "x86_64") == "ubuntu-x64"
    assert InstallPrebuiltLlvm.triton_suffix("manylinux", "aarch64") == "ubuntu-arm64"
    assert InstallPrebuiltLlvm.triton_suffix("macos", "arm64") == "macos-arm64"
    assert InstallPrebuiltLlvm.triton_suffix("macos", "x86_64") == "macos-x64"
    assert InstallPrebuiltLlvm.triton_suffix("windows", "amd64") == "windows-x64"
  end

  test "triton suffix rejects unsupported platforms" do
    assert_raise Mix.Error, ~r/no Triton LLVM archive/, fn ->
      InstallPrebuiltLlvm.triton_suffix("windows", "arm64")
    end
  end

  test "triton metadata URL uses the requested ref" do
    assert InstallPrebuiltLlvm.triton_metadata_url(
             "882eb72e1858bfd588fafa4677b86ce00e9da872",
             "cmake/llvm-info.json"
           ) ==
             "https://raw.githubusercontent.com/triton-lang/triton/882eb72e1858bfd588fafa4677b86ce00e9da872/cmake/llvm-info.json"
  end

  test "installs a tarball and points LLVM_CONFIG_PATH at it" do
    fixture =
      Path.join(System.tmp_dir!(), "beaver-llvm-install-#{System.unique_integer([:positive])}")

    tar = fixture <> ".tar.gz"
    dest = fixture <> "-dest"
    github_env = fixture <> "-env"

    try do
      File.mkdir_p!(Path.join(fixture, "bin"))
      File.write!(Path.join(fixture, "bin/llvm-config"), "#!/bin/sh\necho 17.0.0\n")

      :ok =
        :erl_tar.create(
          String.to_charlist(tar),
          [
            {~c"bin/llvm-config", File.read!(Path.join(fixture, "bin/llvm-config"))}
          ],
          [:compressed]
        )

      output =
        capture_io(fn ->
          InstallPrebuiltLlvm.run([
            "--asset-url",
            "file://#{tar}",
            "--install-dir",
            dest,
            "--github-env",
            github_env
          ])
        end)

      assert output =~ "LLVM_CONFIG_PATH=#{Path.join(dest, "bin/llvm-config")}"
      assert File.exists?(Path.join(dest, "bin/llvm-config"))
      assert File.read!(Path.join(dest, "bin/llvm-config")) =~ "17.0.0"

      exported = File.read!(github_env)
      assert exported =~ "LLVM_CONFIG_PATH=#{Path.join(dest, "bin/llvm-config")}"
      assert exported =~ "LLVM_PREBUILT_DIR=#{dest}"
    after
      File.rm_rf!(fixture)
      File.rm_rf!(dest)
      File.rm(github_env)
      File.rm(tar)
    end
  end

  test "sha256 mismatch aborts the install" do
    fixture =
      Path.join(System.tmp_dir!(), "beaver-llvm-sha-#{System.unique_integer([:positive])}")

    tar = fixture <> ".tar.gz"
    dest = fixture <> "-dest"

    try do
      File.mkdir_p!(Path.join(fixture, "bin"))
      File.write!(Path.join(fixture, "bin/llvm-config"), "#!/bin/sh\n")

      :ok =
        :erl_tar.create(
          String.to_charlist(tar),
          [{~c"bin/llvm-config", "#!/bin/sh\n"}],
          [:compressed]
        )

      assert_raise Mix.Error, ~r/sha256 mismatch/, fn ->
        capture_io(fn ->
          InstallPrebuiltLlvm.run([
            "--asset-url",
            "file://#{tar}",
            "--sha256",
            String.duplicate("0", 64),
            "--install-dir",
            dest
          ])
        end)
      end
    after
      File.rm_rf!(fixture)
      File.rm_rf!(dest)
      File.rm(tar)
    end
  end
end
