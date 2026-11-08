defmodule Beaver.LLVM do
  @moduledoc false

  def asset(metadata, opts) do
    {os, arch} = MLIR.Toolchain.platform(opts)
    artifact = Map.fetch!(metadata["artifacts"], "#{os}_#{arch}")

    %{
      name: "mlir_#{os}_#{arch}_#{metadata["source_ref"]}.zip",
      sha256: artifact["sha256"],
      endpoint: "repos/#{metadata["repo"]}/actions/artifacts/#{artifact["id"]}/zip"
    }
  end

  def install!(metadata, opts) do
    asset = asset(metadata, opts)

    dir =
      Path.expand(
        opts[:install_dir] || System.get_env("LLVM_PREBUILT_DIR") ||
          Path.join(System.tmp_dir!(), "llvm-prebuilt")
      )

    if dir in [Path.expand("/"), System.user_home!(), File.cwd!()],
      do: Mix.raise("protected install directory")

    if File.exists?(dir) and File.ls!(dir) != [] and not File.regular?(config(dir)),
      do: Mix.raise("destination is not an LLVM installation: #{dir}")

    File.mkdir_p!(Path.dirname(dir))
    stage = dir <> ".install-#{System.pid()}-#{System.unique_integer([:positive])}"
    File.mkdir!(stage)

    try do
      root =
        if opts[:from_source],
          do: build!(metadata["source_ref"], stage, opts),
          else: download_or_build!(metadata, asset, stage, opts)

      verify_revision!(root, metadata["source_ref"])

      for path <- Path.wildcard(Path.join(root, "bin/*")),
          File.regular?(path),
          do: File.chmod!(path, 0o755)

      previous = Path.join(stage, "previous")
      if File.exists?(dir), do: File.rename!(dir, previous)

      try do
        File.rename!(root, dir)
      rescue
        error ->
          if File.exists?(previous), do: File.rename!(previous, dir)
          reraise error, __STACKTRACE__
      end

      export_env(dir, metadata["source_ref"], opts[:github_env] || System.get_env("GITHUB_ENV"))

      dir
    after
      File.rm_rf!(stage)
    end
  end

  def export_env(dir, ref, env_file \\ nil) do
    output = """
    LLVM_PREBUILT_DIR=#{dir}
    LLVM_CONFIG_PATH=#{config(dir)}
    LLVM_SOURCE_REF=#{ref}
    """

    IO.write(output)
    if env_file, do: File.write!(env_file, output, [:append])
  end

  defp download_or_build!(metadata, asset, stage, opts) do
    zip = Path.join(stage, "artifact.zip")
    token = opts[:github_token] || System.get_env("GH_TOKEN") || System.get_env("GITHUB_TOKEN")
    env = if token, do: [{"GH_TOKEN", token}], else: []

    result =
      if System.find_executable("gh"),
        do:
          System.cmd("gh", ["api", asset.endpoint], env: env, into: File.stream!(zip, [:write])),
        else: {nil, 1}

    case result do
      {_, 0} ->
        extract!(zip, Path.join(stage, "extracted"), asset.sha256)

      _ ->
        IO.puts("Pinned CI artifact unavailable; building the same LLVM source revision.")
        build!(metadata["source_ref"], stage, opts)
    end
  end

  # Public for fixture tests; check the digest and paths before extracting.
  def extract!(zip, destination, expected) do
    actual =
      zip
      |> File.stream!(1_048_576, [])
      |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
      |> :crypto.hash_final()
      |> Base.encode16(case: :lower)

    unless actual == expected, do: Mix.raise("LLVM artifact SHA-256 mismatch")
    {:ok, entries} = :zip.list_dir(String.to_charlist(zip))

    for entry <- entries, elem(entry, 0) == :zip_file do
      path = entry |> elem(1) |> List.to_string() |> String.replace("\\", "/")

      if Path.type(path) != :relative or ".." in Path.split(path),
        do: Mix.raise("unsafe LLVM artifact path")
    end

    File.mkdir_p!(destination)
    {:ok, _} = :zip.extract(String.to_charlist(zip), cwd: String.to_charlist(destination))
    root = Path.join(destination, "llvm-install")
    unless File.regular?(config(root)), do: Mix.raise("LLVM artifact has no llvm-config")
    root
  end

  def verify_revision!(root, ref) do
    header = File.read!(Path.join(root, "include/llvm/Support/VCSRevision.h"))
    unless String.contains?(header, ref), do: Mix.raise("LLVM artifact source revision mismatch")
    :ok
  end

  defp config(root),
    do:
      Path.join(
        root,
        if(match?({:win32, _}, :os.type()), do: "bin/llvm-config.exe", else: "bin/llvm-config")
      )

  defp build!(ref, stage, opts) do
    source = Path.join(stage, "source")
    build = Path.join(stage, "build")
    install = Path.join(stage, "llvm-install")
    run!("git", ["init", source])

    run!("git", [
      "-C",
      source,
      "fetch",
      "--depth=1",
      "https://github.com/llvm/llvm-project.git",
      ref
    ])

    run!("git", ["-C", source, "checkout", "--detach", "FETCH_HEAD"])

    run!("cmake", [
      "-S",
      Path.join(source, "llvm"),
      "-B",
      build,
      "-G",
      "Ninja",
      "-C",
      Path.expand("../../llvm-source.cmake", __DIR__),
      "-DCMAKE_INSTALL_PREFIX=#{install}"
    ])

    run!("cmake", [
      "--build",
      build,
      "--target",
      "install-mlirdevelopment-distribution",
      "--parallel",
      to_string(opts[:jobs] || 4)
    ])

    install
  end

  def run!(command, args, opts \\ []) do
    case System.cmd(
           command,
           args,
           Keyword.merge([stderr_to_stdout: true, into: IO.stream()], opts)
         ) do
      {_, 0} -> :ok
      {_, status} -> Mix.raise("#{command} exited with #{status}")
    end
  end
end
