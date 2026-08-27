defmodule Beaver.ToolchainMetadata do
  @moduledoc false

  @metadata_path Path.expand("../../../../native-deps.json", __DIR__)

  def path, do: @metadata_path

  def load! do
    @metadata_path
    |> File.read!()
    |> JSON.decode!()
    |> validate!()
  end

  def encode!, do: load!() |> JSON.encode!()

  def llvm_repo!, do: load!() |> get_in(["llvm", "repo"])
  def llvm_tag!, do: load!() |> get_in(["llvm", "tag"])

  def llvm_revision!(os_name, arch) do
    platform = "#{os_name}_#{arch}"

    case load!() |> get_in(["llvm", "default_revisions", platform]) do
      revision when is_binary(revision) -> revision
      nil -> Mix.raise("native-deps.json has no default LLVM revision for #{platform}")
    end
  end

  defp validate!(
         %{
           "schema_version" => 2,
           "kinda" => %{"git_url" => git_url, "ref" => kinda_ref},
           "llvm" => %{
             "repo" => llvm_repo,
             "tag" => llvm_tag,
             "default_revisions" => llvm_revisions
           }
         } = metadata
       )
       when is_binary(git_url) and is_binary(kinda_ref) and is_binary(llvm_repo) and
              is_binary(llvm_tag) and is_map(llvm_revisions) do
    if map_size(llvm_revisions) > 0 and
         Enum.all?(llvm_revisions, fn {platform, revision} ->
           is_binary(platform) and is_binary(revision) and
             revision =~ ~r/^\d{8}\+[0-9a-f]+$/
         end) do
      metadata
    else
      Mix.raise("invalid Beaver toolchain metadata in #{@metadata_path}: #{inspect(metadata)}")
    end
  end

  defp validate!(metadata) do
    Mix.raise("invalid Beaver toolchain metadata in #{@metadata_path}: #{inspect(metadata)}")
  end
end
