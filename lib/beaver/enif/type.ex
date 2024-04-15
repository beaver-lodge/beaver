defmodule Beaver.ENIF.Type do
  @moduledoc """
  Query the MLIR type of an Erlang term or environment.
  """
  alias Beaver.MLIR

  defp query_type(obj, opts) do
    Beaver.Deferred.from_opts(
      opts,
      fn %MLIR.Context{ref: ref} ->
        ref = MLIR.CAPI.mif_raw_mlir_type_of_enif_obj(ref, obj) |> Beaver.Native.check!()
        %MLIR.Type{ref: ref}
      end
    )
  end

  @spec env(Beaver.Deferred.opts()) :: Beaver.Deferred.type()
  def env(opts \\ []) do
    query_type(:env, opts)
  end

  @spec term(Beaver.Deferred.opts()) :: Beaver.Deferred.type()
  def term(opts \\ []) do
    query_type(:term, opts)
  end
end
