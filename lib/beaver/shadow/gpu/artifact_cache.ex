defmodule Beaver.Shadow.GPU.ArtifactCache do
  @moduledoc """
  Deterministic caching of packaged GPU binaries (`gpu.binary`).

  The expensive part of a GPU launch is not the launch itself but the
  `convert-gpu-to-nvvm`/`gpu-module-to-binary` compilation that produces the
  PTX/cubin artifact. `Beaver.Shadow.GPU` runs that compilation once
  per `(source, target)` pair and reuses the packaged module bytecode for every
  subsequent candidate or experiment with the same input and target.

  The key deliberately does not include candidate choices or durations: the
  packaged binary only depends on the source and the target configuration.
  """

  alias Beaver.MLIR
  alias MLIR.CompilationCache
  alias MLIR.CompilationRuntime.CacheKey

  @namespace "beaver_gpu_artifact"

  @doc "Looks up a packaged GPU binary for `source` and `target`."
  @spec get(CompilationCache.cache(), String.t(), MLIR.Attribute.t(), MLIR.Context.t()) ::
          {:ok, MLIR.Module.t()} | :miss | {:error, term()}
  def get(cache, source_key, %MLIR.Attribute{} = target, %MLIR.Context{} = context) do
    key = key(source_key, target)

    case CompilationCache.get(cache, key) do
      {:ok, %{bytecode: bytecode}} ->
        module = MLIR.Module.create!(bytecode, ctx: context)
        {:ok, module}

      :miss ->
        :miss

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Stores the packaged GPU binary for `source` and `target`."
  @spec put(CompilationCache.cache(), String.t(), MLIR.Attribute.t(), MLIR.Module.t()) ::
          :ok | {:error, term()}
  def put(cache, source_key, %MLIR.Attribute{} = target, %MLIR.Module{} = packaged) do
    key = key(source_key, target)
    CompilationCache.put(cache, key, %{bytecode: MLIR.Bytecode.write!(packaged)})
  end

  @doc "Computes the stable cache key from a source key and target."
  @spec key(String.t(), MLIR.Attribute.t()) :: String.t()
  def key(source_key, %MLIR.Attribute{} = target) when is_binary(source_key) do
    target_text = target |> MLIR.to_string()

    CacheKey.digest({@namespace, source_key, target_text})
  end
end
