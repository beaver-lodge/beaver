defmodule Beaver.MLIR.CompilationCache do
  @moduledoc """
  Common access to the incremental compiler's memory and filesystem caches.

  A cache can be `:memory`, `{:memory, server}`, `{:file, directory}`, a memory
  cache server, or a `Beaver.MLIR.CompilationCache.File` value.
  """

  alias __MODULE__.{File, Memory}

  @type cache ::
          :memory
          | {:memory, GenServer.server()}
          | {:file, Path.t()}
          | GenServer.server()
          | File.t()

  @spec get(cache(), String.t()) :: {:ok, map()} | :miss | {:error, term()}
  def get(cache, key), do: dispatch(cache, :get, [key])

  @spec put(cache(), String.t(), map()) :: :ok | {:error, term()}
  def put(cache, key, value), do: dispatch(cache, :put, [key, value])

  @spec delete(cache(), String.t()) :: :ok | {:error, term()}
  def delete(cache, key), do: dispatch(cache, :delete, [key])

  @spec clear(cache()) :: :ok | {:error, term()}
  def clear(cache), do: dispatch(cache, :clear, [])

  defp dispatch(:memory, function, args),
    do: apply(Memory, function, [Memory.default_name() | args])

  defp dispatch({:memory, server}, function, args), do: apply(Memory, function, [server | args])
  defp dispatch({:file, root}, function, args), do: apply(File, function, [File.new(root) | args])
  defp dispatch(%File{} = cache, function, args), do: apply(File, function, [cache | args])
  defp dispatch(server, function, args), do: apply(Memory, function, [server | args])
end
