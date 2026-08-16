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
  def llvm_revision!, do: load!() |> get_in(["llvm", "default_revision"])

  defp validate!(
         %{
           "schema_version" => 1,
           "kinda" => %{"git_url" => git_url, "ref" => kinda_ref},
           "llvm" => %{
             "repo" => llvm_repo,
             "tag" => llvm_tag,
             "default_revision" => llvm_revision
           }
         } = metadata
       )
       when is_binary(git_url) and is_binary(kinda_ref) and is_binary(llvm_repo) and
              is_binary(llvm_tag) and is_binary(llvm_revision) do
    metadata
  end

  defp validate!(metadata) do
    Mix.raise("invalid Beaver toolchain metadata in #{@metadata_path}: #{inspect(metadata)}")
  end
end
