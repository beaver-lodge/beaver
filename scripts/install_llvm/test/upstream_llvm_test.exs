defmodule Beaver.UpstreamLLVMTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureIO

  alias Beaver.UpstreamLLVM
  alias Mix.Tasks.Beaver.{InstallPrebuiltLlvm, ResolveUpstreamLlvm}

  @base "https://api.github.com/repos/llvm/eudsl"
  @runs "#{@base}/actions/workflows/build_llvm.yml/runs?status=completed&per_page=100"
  @run_url "#{@base}/actions/runs/42"
  @jobs "#{@run_url}/jobs?per_page=100&page=1"
  @artifacts "#{@run_url}/artifacts?per_page=100&page=1"
  @source String.duplicate("a", 40)
  @head String.duplicate("b", 40)
  @source_url "#{@base}/contents/third_party/llvm-project?ref=#{@head}"

  setup do
    metadata = Beaver.ToolchainMetadata.load!()["llvm"]

    run = %{
      "id" => 42,
      "status" => "completed",
      "conclusion" => "failure",
      "path" => ".github/workflows/build_llvm.yml",
      "head_sha" => @head,
      "head_branch" => "update-llvm",
      "head_repository" => %{"full_name" => metadata["repo"]}
    }

    artifacts =
      metadata["artifacts"]
      |> Map.keys()
      |> Enum.sort()
      |> Enum.with_index(1)
      |> Enum.map(fn {platform, id} ->
        %{
          "id" => id,
          "name" => "mlir_#{platform}_artifact",
          "expired" => false,
          "digest" => "sha256:" <> String.duplicate(to_string(id), 64)
        }
      end)

    jobs =
      Enum.map(metadata["artifacts"], fn {platform, _} ->
        %{"name" => "build-llvm-mlir #{platform}", "conclusion" => "success"}
      end)

    compare =
      "https://api.github.com/repos/llvm/llvm-project/compare/#{metadata["source_ref"]}...#{@source}"

    api = %{
      @runs => %{"workflow_runs" => [run]},
      @run_url => run,
      @source_url => %{
        "type" => "submodule",
        "sha" => @source,
        "submodule_git_url" => "https://github.com/llvm/llvm-project.git"
      },
      compare => %{"status" => "ahead"},
      @jobs => %{"jobs" => jobs},
      @artifacts => %{"artifacts" => artifacts}
    }

    %{metadata: metadata, api: api, run: run, compare: compare}
  end

  test "freezes four native artifacts despite unrelated downstream failures", context do
    manifest = resolve(context)
    assert manifest["build_run"] == 42
    assert manifest["source_ref"] == @source
    assert manifest["head_sha"] == @head
    assert Map.keys(manifest["artifacts"]) == Map.keys(context.metadata["artifacts"])

    for {platform, pin} <- manifest["artifacts"] do
      assert pin["id"] > 0
      assert byte_size(pin["sha256"]) == 64
      assert Beaver.LLVM.asset(manifest, platform_options(platform)).sha256 == pin["sha256"]
    end

    assert manifest == resolve(context, "42")

    assert UpstreamLLVM.validate!(JSON.decode!(JSON.encode!(manifest)), context.metadata) ==
             manifest
  end

  test "latest ignores fork, unrelated branch, wrong workflow and incomplete runs", context do
    invalid = [
      put_in(context.run, ["head_repository", "full_name"], "other/eudsl"),
      %{context.run | "head_branch" => "unrelated-pr"},
      %{context.run | "path" => ".github/workflows/other.yml"},
      %{context.run | "status" => "in_progress"}
    ]

    api = put_in(context.api, [@runs, "workflow_runs"], invalid ++ [context.run])
    assert resolve(%{context | api: api})["build_run"] == 42

    for run <- invalid do
      api = put_in(context.api, [@run_url], run)

      assert_raise Mix.Error, ~r/untrusted or incomplete/, fn ->
        resolve(%{context | api: api}, "42")
      end
    end

    api = put_in(context.api, [@runs, "workflow_runs"], invalid)
    assert_raise Mix.Error, ~r/no completed/, fn -> resolve(%{context | api: api}) end
  end

  test "rejects invalid run IDs and API identity mismatches", context do
    for id <- ["0", "-1", "42/../43", "42\n", "abc"] do
      assert_raise Mix.Error, ~r/positive GitHub/, fn -> resolve(context, id) end
    end

    api = put_in(context.api, [@run_url, "id"], 43)
    assert_raise Mix.Error, ~r/build ID mismatch/, fn -> resolve(%{context | api: api}, "42") end
  end

  test "rejects a regressed or divergent LLVM source instead of falling back", context do
    for status <- ["behind", "diverged"] do
      api = put_in(context.api, [context.compare, "status"], status)

      assert_raise Mix.Error, ~r/must contain Beaver's pinned/, fn ->
        resolve(%{context | api: api})
      end
    end

    for {key, value} <- [
          {"type", "file"},
          {"sha", "short"},
          {"submodule_git_url", "https://github.com/other/llvm-project.git"}
        ] do
      api = put_in(context.api, [@source_url, key], value)
      assert_raise Mix.Error, ~r/source identity/, fn -> resolve(%{context | api: api}) end
    end
  end

  test "a failed native platform blocks the whole manifest", context do
    for status <- ["failure", "cancelled", "skipped", nil] do
      jobs = Enum.map(context.api[@jobs]["jobs"], &Map.put(&1, "conclusion", status))
      api = put_in(context.api, [@jobs, "jobs"], jobs)
      assert_raise Mix.Error, ~r/did not succeed/, fn -> resolve(%{context | api: api}) end
    end
  end

  test "requires unique, unexpired artifacts with full SHA-256 digests", context do
    [artifact | rest] = context.api[@artifacts]["artifacts"]

    for artifacts <- [
          rest,
          [artifact, artifact | rest],
          [%{artifact | "expired" => true} | rest],
          [%{artifact | "digest" => nil} | rest],
          [%{artifact | "digest" => "sha256:short"} | rest],
          [%{artifact | "digest" => "sha256:" <> String.duplicate("a", 63) <> "\n"} | rest],
          [%{artifact | "id" => -1} | rest]
        ] do
      api = put_in(context.api, [@artifacts, "artifacts"], artifacts)
      assert_raise Mix.Error, fn -> resolve(%{context | api: api}) end
    end
  end

  test "paginates jobs and artifacts instead of overlooking later pages", context do
    api =
      context.api
      |> Map.put("#{@run_url}/jobs?per_page=100&page=2", context.api[@jobs])
      |> Map.put("#{@run_url}/artifacts?per_page=100&page=2", context.api[@artifacts])
      |> put_in([@jobs, "jobs"], List.duplicate(%{"name" => "downstream"}, 100))
      |> put_in([@artifacts, "artifacts"], List.duplicate(%{"name" => "other"}, 100))

    assert resolve(%{context | api: api}) == resolve(context)
  end

  test "manifest validation rejects wrong repos, revisions and platform coverage", context do
    manifest = resolve(context)

    for invalid <- [
          nil,
          %{manifest | "schema_version" => 0},
          %{manifest | "repo" => "other/eudsl"},
          %{manifest | "source_ref" => "short"},
          %{manifest | "head_sha" => "short"},
          %{manifest | "build_run" => 0},
          %{manifest | "artifacts" => %{}},
          put_in(manifest, ["artifacts", "windows_amd64", "sha256"], "bad"),
          put_in(manifest, ["artifacts", "windows_amd64", "id"], -1)
        ] do
      assert_raise Mix.Error, ~r/invalid upstream LLVM CI manifest/, fn ->
        UpstreamLLVM.validate!(invalid, context.metadata)
      end
    end
  end

  @tag :tmp_dir
  test "Mix resolver writes one frozen JSON output for the entire matrix", context do
    output = Path.join(context.tmp_dir, "github-output")
    File.write!(output, "existing=keep\n")

    json =
      capture_io(fn ->
        ResolveUpstreamLlvm.run(["--build-run", "42", "--github-output", output],
          get: &Map.fetch!(context.api, &1)
        )
      end)
      |> String.trim()

    assert JSON.decode!(json) == resolve(context)
    assert File.read!(output) == "existing=keep\nllvm_ci_manifest=#{json}\n"
  end

  test "installer resolves the frozen ZIP without rediscovery for Linux and Windows", context do
    manifest = resolve(context)

    for platform <- ["manylinux_x86_64", "windows_amd64"] do
      opts = platform_options(platform)

      output =
        capture_io(fn ->
          InstallPrebuiltLlvm.run([
            "--ci-manifest",
            JSON.encode!(manifest),
            "--resolve-only",
            "--asset-os",
            opts[:asset_os],
            "--asset-arch",
            opts[:asset_arch]
          ])
        end)

      assert output =~ "mlir_#{platform}_#{@source}.zip"
      assert output =~ "/actions/artifacts/#{manifest["artifacts"][platform]["id"]}/zip"
      assert output =~ "LLVM_ARTIFACT_SHA256=#{manifest["artifacts"][platform]["sha256"]}"
    end
  end

  test "installer rejects mixed CI and release/source selectors", context do
    args = ["--ci-manifest", JSON.encode!(resolve(context)), "--resolve-only"]

    for conflicting <- [["--asset-revision", "latest"], ["--from-source"]] do
      assert_raise Mix.Error, ~r/cannot be combined/, fn ->
        InstallPrebuiltLlvm.run(args ++ conflicting)
      end
    end

    output =
      capture_io(fn ->
        InstallPrebuiltLlvm.run([
          "--ci-manifest",
          "",
          "--asset-url",
          "https://example.test/llvm.tar.gz",
          "--resolve-only"
        ])
      end)

    assert output =~ "LLVM_PREBUILT_URL=https://example.test/llvm.tar.gz"
  end

  @tag :tmp_dir
  test "CI installation publishes its digest and never falls back when download fails", context do
    config = if match?({:win32, _}, :os.type()), do: "llvm-config.exe", else: "llvm-config"
    zip = Path.join(context.tmp_dir, "upstream.zip")
    install_dir = Path.join(context.tmp_dir, "installed")
    env_file = Path.join(context.tmp_dir, "env")

    files = [
      {String.to_charlist("llvm-install/bin/#{config}"), "binary"},
      {~c"llvm-install/include/llvm/Support/VCSRevision.h", "#define LLVM_REVISION \"#{@source}\""}
    ]

    {:ok, _} = :zip.create(String.to_charlist(zip), files)
    sha = :crypto.hash(:sha256, File.read!(zip)) |> Base.encode16(case: :lower)

    manifest =
      resolve(context)
      |> Map.update!("artifacts", fn artifacts ->
        Map.new(artifacts, fn {platform, pin} -> {platform, %{pin | "sha256" => sha}} end)
      end)

    opts = [install_dir: install_dir, github_env: env_file, require_artifact: true]

    download = fn asset, destination, _token ->
      assert asset.sha256 == sha
      File.cp!(zip, destination)
      {nil, 0}
    end

    output =
      capture_io(fn ->
        assert Beaver.LLVM.install!(manifest, Keyword.put(opts, :download, download)) ==
                 install_dir
      end)

    assert output =~ "LLVM_ARTIFACT_SHA256=#{sha}"
    assert output == File.read!(env_file)
    assert File.read!(Path.join([install_dir, "bin", config])) == "binary"

    assert_raise Mix.Error, ~r/refusing source fallback/, fn ->
      Beaver.LLVM.install!(manifest, Keyword.put(opts, :download, fn _, _, _ -> {nil, 1} end))
    end

    assert File.read!(Path.join([install_dir, "bin", config])) == "binary"
    assert File.read!(env_file) == output
    assert Path.wildcard(install_dir <> ".install-*") == []
  end

  defp resolve(context, run \\ "latest"),
    do: UpstreamLLVM.resolve!(context.metadata, run, get: &Map.fetch!(context.api, &1))

  defp platform_options("windows_amd64"), do: [asset_os: "windows", asset_arch: "amd64"]
  defp platform_options("macos_arm64"), do: [asset_os: "macos", asset_arch: "arm64"]

  defp platform_options("manylinux_" <> arch),
    do: [asset_os: "manylinux", asset_arch: arch]
end
