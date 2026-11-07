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

  defp validate!(
         %{
           "schema_version" => 3,
           "kinda" => %{"ref" => ref},
           "llvm" => %{"source_ref" => llvm_ref, "artifacts" => artifacts}
         } = metadata
       ) do
    unless is_binary(ref) and is_binary(llvm_ref) and llvm_ref =~ ~r/^[0-9a-f]{40}$/ and
             is_map(artifacts) and map_size(artifacts) == 4 and
             Enum.all?(artifacts, fn {_, artifact} ->
               is_integer(artifact["id"]) and is_binary(artifact["sha256"]) and
                 artifact["sha256"] =~ ~r/^[0-9a-f]{64}$/
             end),
           do: Mix.raise("invalid toolchain source or artifact pin")

    metadata
  end

  defp validate!(_), do: Mix.raise("native-deps.json must use the current toolchain schema")
end
