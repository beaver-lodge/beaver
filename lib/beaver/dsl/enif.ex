defmodule Beaver.ENIF do
  use Beaver
  require Beaver.MLIR.Dialect.Func
  alias Beaver.MLIR.Dialect.Func
  alias MLIR.Type

  defp mlir_t({ref, _size}) when is_reference(ref) do
    %MLIR.Type{ref: ref}
  end

  @doc """
  insert external functions of ENIF into current MLIR block
  """
  def populate_external_functions(ctx, block) do
    mlir ctx: ctx, block: block do
      for {name, arg_types, ret_type} <- MLIR.CAPI.mif_raw_enif_signatures(ctx.ref) do
        Func.func _(
                    sym_name: "\"#{name}\"",
                    sym_visibility: MLIR.Attribute.string("private"),
                    function_type:
                      Type.function(Enum.map(arg_types, &mlir_t/1), [mlir_t(ret_type)])
                  ) do
          region do
          end
        end
      end
    end
  end

  defmodule ErlNifEnv do
    def mlir_t(opts \\ []) do
      Beaver.Deferred.from_opts(
        opts,
        fn %MLIR.Context{ref: ref} ->
          %MLIR.Type{ref: MLIR.CAPI.mif_raw_mlir_type_ErlNifEnv(ref)}
        end
      )
    end
  end

  defmodule ERL_NIF_TERM do
    def mlir_t(opts \\ []) do
      Beaver.Deferred.from_opts(
        opts,
        fn %MLIR.Context{ref: ref} ->
          %MLIR.Type{ref: MLIR.CAPI.mif_raw_mlir_type_ERL_NIF_TERM(ref)}
        end
      )
    end
  end
end
