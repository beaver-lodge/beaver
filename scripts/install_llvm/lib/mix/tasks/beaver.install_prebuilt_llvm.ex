defmodule Mix.Tasks.Beaver.InstallPrebuiltLlvm do
  @moduledoc """
  Installs a prebuilt LLVM/MLIR distribution and prints `LLVM_CONFIG_PATH`.

  By default it downloads the matching `llvm/eudsl` nightly build for the
  current platform. `--triton` installs the exact LLVM archive Triton pins
  (resolved from `triton-lang/triton`'s `llvm-info.json` at run time). Any URL
  can be used instead, for example Triton's pinned LLVM archive:

      (cd scripts/install_llvm && mix beaver.install_prebuilt_llvm \\
        --asset-url https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-b010a18d-ubuntu-x64-1.tar.gz \\
        --sha256 <sha256>)

  The task lives in its own tiny Mix project (`scripts/install_llvm`) so
  running it never compiles Beaver or its NIF.

  ## Options

  Each option has an environment variable fallback used when the flag is not
  given:

    * `--install-dir PATH` / `LLVM_PREBUILT_DIR` — where to install; defaults
      to `$RUNNER_TEMP`/`$TMPDIR`/`/tmp` + `llvm-prebuilt`. A leading
      positional argument is accepted for compatibility with the old shell
      wrapper.
    * `--repo REPO` / `LLVM_EUDSL_REPO` — defaults to `llvm/eudsl`.
    * `--tag TAG` / `LLVM_EUDSL_TAG` — defaults to `llvm`.
    * `--asset-revision REV` / `LLVM_EUDSL_ASSET_REVISION` — defaults to
      `20260804+eb50d8775`; `latest` resolves the newest asset through the
      GitHub API.
    * `--asset-name NAME` / `LLVM_EUDSL_ASSET_NAME` — exact asset file name.
    * `--asset-url URL` / `LLVM_EUDSL_ASSET_URL` — download from an arbitrary
      URL (the asset name defaults to the URL basename).
    * `--asset-os OS` / `LLVM_EUDSL_ASSET_OS` and `--asset-arch ARCH` /
      `LLVM_EUDSL_ASSET_ARCH` — override platform detection for asset naming.
    * `--triton` / `LLVM_TRITON=1` — install the prebuilt LLVM Triton pins
      (`cmake/llvm-build-info.json` + `cmake/llvm-info.json` from
      `triton-lang/triton`); the archive URL and sha256 are derived from those
      files, so the pin tracks Triton's LLVM bumps.
    * `--sha256 DIGEST` / `LLVM_EUDSL_SHA256` — verify the downloaded archive.
    * `--github-token TOKEN` / `GITHUB_TOKEN` or `GH_TOKEN` — used for
      `latest` resolution.
    * `--github-env PATH` / `GITHUB_ENV` — file to append the exported
      variables to. Defaults to the `GITHUB_ENV` environment variable; pass an
      explicit path when running in an environment that sets `GITHUB_ENV` but
      should not be mutated (e.g. tests).
    * `--resolve-only` / `LLVM_EUDSL_RESOLVE_ONLY=1` — print
      `LLVM_PREBUILT_ASSET_NAME` and `LLVM_PREBUILT_URL` without downloading.

  When `GITHUB_ENV` is set, `LLVM_CONFIG_PATH`, `LLVM_PREBUILT_DIR`,
  `LLVM_PREBUILT_ASSET_NAME` and `LLVM_PREBUILT_URL` are appended to it so
  subsequent CI steps can pick up the installation.
  """

  use Mix.Task

  @shortdoc "Installs a prebuilt LLVM/MLIR distribution"

  @default_revision "20260804+eb50d8775"

  @switches [
    install_dir: :string,
    repo: :string,
    tag: :string,
    asset_revision: :string,
    asset_name: :string,
    asset_url: :string,
    asset_os: :string,
    asset_arch: :string,
    sha256: :string,
    github_token: :string,
    github_env: :string,
    triton: :boolean,
    resolve_only: :boolean
  ]

  @impl Mix.Task
  def run(args) do
    {opts, positional} = OptionParser.parse!(args, strict: @switches)
    opts = opts |> reject_empty() |> merge_env()

    install_dir = opts[:install_dir] || List.first(positional) || default_install_dir()
    {os_name, arch} = platform(opts)
    {asset_name, asset_url, sha256} = resolve_asset(opts, os_name, arch)

    if opts[:resolve_only] do
      IO.puts("LLVM_PREBUILT_ASSET_NAME=#{asset_name}")
      IO.puts("LLVM_PREBUILT_URL=#{asset_url}")
    else
      install!(install_dir, asset_name, asset_url, sha256, opts[:github_env])
    end
  end

  defp merge_env(opts) do
    opts
    |> put_default(:repo, "llvm/eudsl", "LLVM_EUDSL_REPO")
    |> put_default(:tag, "llvm", "LLVM_EUDSL_TAG")
    |> put_default(:asset_revision, @default_revision, "LLVM_EUDSL_ASSET_REVISION")
    |> put_env(:install_dir, "LLVM_PREBUILT_DIR")
    |> put_env(:asset_name, "LLVM_EUDSL_ASSET_NAME")
    |> put_env(:asset_url, "LLVM_EUDSL_ASSET_URL")
    |> put_env(:asset_os, "LLVM_EUDSL_ASSET_OS")
    |> put_env(:asset_arch, "LLVM_EUDSL_ASSET_ARCH")
    |> put_env(:sha256, "LLVM_EUDSL_SHA256")
    |> put_env(:github_env, "GITHUB_ENV")
    |> put_env(:github_token, "GITHUB_TOKEN")
    |> put_env(:github_token, "GH_TOKEN")
    |> maybe_resolve_only()
    |> maybe_triton()
  end

  defp reject_empty(opts) do
    Enum.reject(opts, fn {_key, value} -> value == "" end)
  end

  defp put_default(opts, key, default, env) do
    if Keyword.has_key?(opts, key) do
      opts
    else
      case System.get_env(env) do
        value when value in [nil, ""] -> Keyword.put(opts, key, default)
        value -> Keyword.put(opts, key, value)
      end
    end
  end

  defp put_env(opts, key, env) do
    if Keyword.has_key?(opts, key) do
      opts
    else
      case System.get_env(env) do
        value when value in [nil, ""] -> opts
        value -> Keyword.put(opts, key, value)
      end
    end
  end

  defp maybe_resolve_only(opts) do
    if System.get_env("LLVM_EUDSL_RESOLVE_ONLY") == "1" do
      Keyword.put_new(opts, :resolve_only, true)
    else
      opts
    end
  end

  defp maybe_triton(opts) do
    if System.get_env("LLVM_TRITON") == "1" do
      Keyword.put_new(opts, :triton, true)
    else
      opts
    end
  end

  defp default_install_dir do
    base = System.get_env("RUNNER_TEMP") || System.get_env("TMPDIR") || "/tmp"
    Path.join(base, "llvm-prebuilt")
  end

  defp platform(opts) do
    os_name = opts[:asset_os] || detect_os()
    arch = opts[:asset_arch] || detect_arch(os_name)
    {os_name, arch}
  end

  defp detect_os do
    case :os.type() do
      {:unix, :darwin} -> "macos"
      {:unix, _} -> "manylinux"
      {:win32, _} -> "windows"
      other -> Mix.raise("unsupported operating system: #{inspect(other)}")
    end
  end

  defp detect_arch("windows"), do: "amd64"

  defp detect_arch(os_name) do
    arch = :erlang.system_info(:system_architecture) |> List.to_string()

    arm? = String.starts_with?(arch, "aarch64") or String.starts_with?(arch, "arm64")

    case {os_name, arm?} do
      {"macos", true} -> "arm64"
      {"macos", false} -> "x86_64"
      {_, true} -> "aarch64"
      {_, false} -> "x86_64"
    end
  end

  defp resolve_asset(opts, os_name, arch) do
    cond do
      opts[:triton] ->
        resolve_triton_asset!(opts, os_name, arch)

      is_binary(opts[:asset_url]) ->
        {opts[:asset_name] || Path.basename(opts[:asset_url]), opts[:asset_url], opts[:sha256]}

      true ->
        name =
          opts[:asset_name] ||
            case opts[:asset_revision] do
              "latest" -> resolve_latest_asset!(opts, os_name, arch)
              revision -> "mlir_#{os_name}_#{arch}_#{revision}.tar.gz"
            end

        url = "https://github.com/#{opts[:repo]}/releases/download/#{opts[:tag]}/#{name}"
        {name, url, opts[:sha256]}
    end
  end

  defp resolve_triton_asset!(opts, os_name, arch) do
    ensure_http!()
    suffix = triton_suffix(os_name, arch)
    build_info = triton_json!("cmake/llvm-build-info.json")
    info = triton_json!("cmake/llvm-info.json")

    hash = info["llvm_hash"] || build_info["llvm_hash"]
    build_number = info["build_number"] || build_info["build_number"]
    sha256 = info["sha256sum"][suffix] || opts[:sha256]

    unless is_binary(hash) and is_integer(build_number) do
      Mix.raise(
        "invalid Triton LLVM info: hash=#{inspect(hash)} build_number=#{inspect(build_number)}"
      )
    end

    name = "llvm-#{String.slice(hash, 0, 8)}-#{suffix}-#{build_number}.tar.gz"
    url = "https://oaitriton.blob.core.windows.net/public/llvm-builds/#{name}"
    {name, url, sha256}
  end

  @doc false
  def triton_suffix(os_name, arch) do
    case {os_name, arch} do
      {"manylinux", "x86_64"} -> "ubuntu-x64"
      {"manylinux", "aarch64"} -> "ubuntu-arm64"
      {"macos", "arm64"} -> "macos-arm64"
      {"macos", "x86_64"} -> "macos-x64"
      {"windows", "amd64"} -> "windows-x64"
      other -> Mix.raise("no Triton LLVM archive for #{inspect(other)}")
    end
  end

  defp triton_json!(path) do
    github_get!("https://raw.githubusercontent.com/triton-lang/triton/main/#{path}", nil)
  end

  defp resolve_latest_asset!(opts, os_name, arch) do
    ensure_http!()
    prefix = "mlir_#{os_name}_#{arch}_"
    release = github_get!(release_url(opts), opts[:github_token])
    assets = fetch_all_assets!(release["id"], opts)

    matches =
      Enum.filter(assets, fn asset ->
        String.starts_with?(asset["name"], prefix) and
          String.ends_with?(asset["name"], ".tar.gz")
      end)

    case matches
         |> Enum.sort_by(fn asset -> {asset["updated_at"] || "", asset["name"]} end)
         |> List.last() do
      nil ->
        Mix.raise("no llvm/eudsl asset found for #{os_name}/#{arch} under tag #{opts[:tag]}")

      asset ->
        asset["name"]
    end
  end

  defp release_url(opts) do
    "https://api.github.com/repos/#{opts[:repo]}/releases/tags/#{opts[:tag]}"
  end

  defp fetch_all_assets!(release_id, opts) do
    Enum.reduce_while(1..10, [], fn page, acc ->
      url =
        "https://api.github.com/repos/#{opts[:repo]}/releases/#{release_id}/assets?per_page=100&page=#{page}"

      case github_get!(url, opts[:github_token]) do
        [] -> {:halt, acc}
        assets -> {:cont, acc ++ assets}
      end
    end)
  end

  defp github_get!(url, token) do
    headers =
      [
        {~c"Accept", ~c"application/vnd.github+json"},
        {~c"User-Agent", ~c"beaver-install-prebuilt-llvm"},
        {~c"X-GitHub-Api-Version", ~c"2022-11-28"}
      ]

    headers =
      case token do
        nil -> headers
        token -> [{~c"Authorization", String.to_charlist("Bearer #{token}")} | headers]
      end

    http_opts = [timeout: 30_000, connect_timeout: 15_000]

    case :httpc.request(:get, {String.to_charlist(url), headers}, http_opts, body_format: :binary) do
      {:ok, {{_, 200, _}, _, body}} ->
        JSON.decode!(body)

      {:ok, {{_, status, _}, _, body}} ->
        Mix.raise("GitHub API #{status} for #{url}: #{inspect(body)}")

      {:error, reason} ->
        Mix.raise("GitHub API request failed for #{url}: #{inspect(reason)}")
    end
  end

  defp ensure_http! do
    {:ok, _} = Application.ensure_all_started(:inets)
    {:ok, _} = Application.ensure_all_started(:ssl)
  end

  defp install!(install_dir, asset_name, asset_url, sha256, github_env) do
    guard_install_dir!(install_dir)
    tmp_root = Path.join(System.tmp_dir!(), "llvm-prebuilt-#{System.unique_integer([:positive])}")
    archive = Path.join(tmp_root, asset_name)
    extract_dir = Path.join(tmp_root, "extract")

    try do
      File.mkdir_p!(tmp_root)
      download!(asset_url, archive)
      verify_sha256!(archive, sha256)
      extract!(archive, extract_dir)

      llvm_config = find_llvm_config!(extract_dir, asset_name)

      asset_root = llvm_config |> Path.dirname() |> Path.dirname()

      File.rm_rf!(install_dir)
      File.mkdir_p!(install_dir)
      copy_tree!(asset_root, install_dir)

      llvm_config_path = Path.join(install_dir, "bin/#{Path.basename(llvm_config)}")
      File.chmod(llvm_config_path, 0o755)

      IO.puts("Installed #{asset_name} into #{install_dir}")
      IO.puts("LLVM_CONFIG_PATH=#{llvm_config_path}")
      write_github_env!(install_dir, asset_name, asset_url, github_env)
    after
      File.rm_rf!(tmp_root)
    end
  end

  defp guard_install_dir!(install_dir) do
    if install_dir in ["", "/"] do
      Mix.raise("refusing to install LLVM into '#{install_dir}'")
    end
  end

  defp download!(url, path) do
    IO.puts("Downloading #{url}")

    cond do
      System.find_executable("curl") ->
        run_download(
          [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--output",
            path,
            "--url",
            url
          ],
          url
        )

      System.find_executable("wget") ->
        run_download(["wget", "--quiet", "--output-document", path, url], url)

      true ->
        Mix.raise("curl or wget is required to download #{url}")
    end
  end

  defp run_download(command, url) do
    {output, status} = System.cmd(hd(command), tl(command), stderr_to_stdout: true)

    unless status == 0 do
      Mix.raise("download failed with status #{status} for #{url}:\n#{output}")
    end

    :ok
  end

  defp verify_sha256!(_path, nil), do: :ok

  defp verify_sha256!(path, expected) do
    digest =
      path
      |> File.stream!(1_048_576, [])
      |> Enum.reduce(:crypto.hash_init(:sha256), fn chunk, ctx ->
        :crypto.hash_update(ctx, chunk)
      end)
      |> :crypto.hash_final()
      |> Base.encode16(case: :lower)

    unless digest == String.downcase(expected) do
      Mix.raise("sha256 mismatch for #{path}: expected #{expected}, got #{digest}")
    end

    :ok
  end

  defp copy_tree!(src_root, dest_root) do
    File.mkdir_p!(dest_root)

    Enum.each(File.ls!(src_root), fn entry ->
      src = Path.join(src_root, entry)
      dest = Path.join(dest_root, entry)

      case File.lstat(src) do
        {:ok, %File.Stat{type: :symlink}} ->
          {:ok, target} = File.read_link(src)
          File.ln_s(target, dest)

        {:ok, %File.Stat{type: :directory}} ->
          copy_tree!(src, dest)

        _ ->
          File.cp!(src, dest)
      end
    end)
  end

  defp extract!(archive, extract_dir) do
    File.mkdir_p!(extract_dir)

    if match?({:win32, _}, :os.type()) do
      # Windows runners do not guarantee a usable system tar for these
      # archives, so go through erl_tar for the Windows prebuilt. The Unix
      # archives cannot use erl_tar: they store negative base-256 mtime fields
      # (reproducible builds) that erl_tar rejects with :integer_overflow, and
      # they rely on symlinks that the system tar handles faithfully.
      case :erl_tar.extract(String.to_charlist(archive), [
             :compressed,
             {:cwd, String.to_charlist(extract_dir)}
           ]) do
        :ok -> :ok
        {:error, reason} -> Mix.raise("failed to extract #{archive}: #{inspect(reason)}")
      end
    else
      {output, status} =
        System.cmd("tar", ["-xzf", archive, "-C", extract_dir], stderr_to_stdout: true)

      unless status == 0 do
        Mix.raise("failed to extract #{archive}:\n#{output}")
      end

      :ok
    end
  end

  # Path.wildcard/1 has Windows-specific separator semantics that are easy to
  # get wrong for a pattern built from a native path, so locate llvm-config by
  # walking the extracted tree instead.
  defp find_llvm_config!(extract_dir, asset_name) do
    case walk_for_llvm_config(extract_dir) do
      nil -> Mix.raise("could not locate bin/llvm-config in #{asset_name}")
      path -> path
    end
  end

  defp walk_for_llvm_config(dir) do
    Enum.find_value(File.ls!(dir), fn entry ->
      path = Path.join(dir, entry)

      cond do
        File.dir?(path) ->
          walk_for_llvm_config(path)

        Path.basename(dir) == "bin" and String.starts_with?(entry, "llvm-config") ->
          path

        true ->
          nil
      end
    end)
  end

  defp write_github_env!(install_dir, asset_name, asset_url, github_env) do
    case github_env || System.get_env("GITHUB_ENV") do
      nil ->
        :ok

      path ->
        content = """
        LLVM_CONFIG_PATH=#{Path.join(install_dir, "bin/llvm-config")}
        LLVM_PREBUILT_DIR=#{install_dir}
        LLVM_PREBUILT_ASSET_NAME=#{asset_name}
        LLVM_PREBUILT_URL=#{asset_url}
        """

        File.write!(path, content, [:append])
    end
  end
end
