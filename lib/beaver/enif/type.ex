defmodule Beaver.ENIF.Type do
  @moduledoc """
  Query the MLIR type of an Erlang term or environment.
  """
  alias Beaver.MLIR

  @typedoc """
  A tuple representing the signature of an enif function.
  Each tuple contains:
  - The name of the function, which is an atom.
  - A list of argument MLIR types.
  - A list of return MLIR types.
  """
  @type signature() :: {atom(), [MLIR.Type.t()], [MLIR.Type.t()]}
  defp query_type(obj, opts) do
    Beaver.Deferred.from_opts(
      opts,
      fn %MLIR.Context{ref: ref} ->
        ref = MLIR.CAPI.beaver_raw_mlir_type_of_enif_obj(ref, obj)
        %MLIR.Type{ref: ref}
      end
    )
  end

  for t <- ~w{env term binary pid} do
    def unquote(:"#{t}")(opts \\ []) do
      query_type(unquote(:"#{t}"), opts)
    end
  end
end
