defmodule Beaver.UpstreamLLVM do
  @moduledoc "Freeze one upstream LLVM CI build into a checksum-pinned installation manifest."

  @workflow ".github/workflows/build_llvm.yml"
  @branches ["main", "update-llvm"]

  def resolve!(metadata, build_run \\ "latest", opts \\ []) do
    token = opts[:github_token] || System.get_env("GH_TOKEN") || System.get_env("GITHUB_TOKEN")
    get = opts[:get] || (&MLIR.Toolchain.github_get!(&1, token))
    repo = metadata["repo"]
    base = "https://api.github.com/repos/#{repo}"
    run = select_run!(get, base, repo, build_run)

    unless trusted_run?(run, repo), do: Mix.raise("untrusted or incomplete upstream LLVM build")

    source_ref = source_ref!(get, base, run["head_sha"], metadata["source_ref"])
    jobs = pages(get, "#{base}/actions/runs/#{run["id"]}/jobs", "jobs")
    artifacts = pages(get, "#{base}/actions/runs/#{run["id"]}/artifacts", "artifacts")

    selected =
      Map.new(metadata["artifacts"], fn {platform, _pin} ->
        {platform, artifact!(platform, jobs, artifacts)}
      end)

    %{
      "schema_version" => 1,
      "repo" => repo,
      "build_run" => run["id"],
      "head_sha" => run["head_sha"],
      "source_ref" => source_ref,
      "artifacts" => selected
    }
    |> validate!(metadata)
  end

  defp source_ref!(get, base, head, pinned) do
    source = get.("#{base}/contents/third_party/llvm-project?ref=#{head}")

    unless source["type"] == "submodule" and
             source["submodule_git_url"] == "https://github.com/llvm/llvm-project.git" and
             digest?(source["sha"], 40),
           do: Mix.raise("upstream build has no verified LLVM source identity")

    comparison =
      get.("https://api.github.com/repos/llvm/llvm-project/compare/#{pinned}...#{source["sha"]}")

    unless comparison["status"] in ["identical", "ahead"],
      do: Mix.raise("upstream LLVM must contain Beaver's pinned source revision")

    source["sha"]
  end

  defp artifact!(platform, jobs, artifacts) do
    unless Enum.any?(jobs, fn job ->
             job["name"] == "build-llvm-mlir #{platform}" and job["conclusion"] == "success"
           end),
           do: Mix.raise("upstream LLVM build did not succeed for #{platform}")

    case Enum.filter(artifacts, &(&1["name"] == "mlir_#{platform}_artifact")) do
      [%{"expired" => false, "digest" => "sha256:" <> sha, "id" => id}]
      when is_integer(id) and id > 0 ->
        unless digest?(sha, 64), do: Mix.raise("invalid upstream artifact SHA-256")
        %{"id" => id, "sha256" => sha}

      _ ->
        Mix.raise("missing, expired or ambiguous upstream LLVM artifact for #{platform}")
    end
  end

  def validate!(manifest, metadata) do
    unless is_map(manifest) and manifest["schema_version"] == 1 and
             manifest["repo"] == metadata["repo"] and
             is_integer(manifest["build_run"]) and manifest["build_run"] > 0 and
             digest?(manifest["source_ref"], 40) and digest?(manifest["head_sha"], 40) and
             valid_artifacts?(manifest["artifacts"], metadata["artifacts"]),
           do: Mix.raise("invalid upstream LLVM CI manifest")

    manifest
  end

  defp valid_artifacts?(artifacts, expected) when is_map(artifacts) do
    Enum.sort(Map.keys(artifacts)) == Enum.sort(Map.keys(expected)) and
      Enum.all?(artifacts, fn
        {_platform, %{"id" => id, "sha256" => sha}} ->
          is_integer(id) and id > 0 and digest?(sha, 64)

        _ ->
          false
      end)
  end

  defp valid_artifacts?(_, _), do: false

  defp select_run!(get, base, repo, "latest") do
    response =
      get.("#{base}/actions/workflows/build_llvm.yml/runs?status=completed&per_page=100")

    Enum.find(response["workflow_runs"], &trusted_run?(&1, repo)) ||
      Mix.raise("no completed upstream LLVM build found")
  end

  defp select_run!(get, base, _repo, id) do
    unless is_binary(id) and id =~ ~r/\A[1-9][0-9]*\z/,
      do: Mix.raise("build-run must be latest or a positive GitHub Actions run ID")

    run = get.("#{base}/actions/runs/#{id}")
    unless run["id"] == String.to_integer(id), do: Mix.raise("upstream LLVM build ID mismatch")
    run
  end

  defp trusted_run?(run, repo) do
    is_integer(run["id"]) and run["id"] > 0 and
      run["status"] == "completed" and run["path"] == @workflow and
      run["head_branch"] in @branches and
      get_in(run, ["head_repository", "full_name"]) == repo and
      digest?(run["head_sha"], 40)
  end

  defp pages(get, url, key, page \\ 1) do
    items = get.("#{url}?per_page=100&page=#{page}") |> Map.fetch!(key)
    if length(items) < 100, do: items, else: items ++ pages(get, url, key, page + 1)
  end

  defp digest?(value, size) when is_binary(value),
    do: byte_size(value) == size and value =~ ~r/\A[0-9a-f]+\z/

  defp digest?(_, _), do: false
end
