defmodule Beaver.ENIF do
  use Beaver
  require Beaver.MLIR.Dialect.Func
  alias Beaver.MLIR.Dialect.Func
  alias MLIR.Type

  defp mlir_t({"c_int", size}) do
    Type.i(size * 8)
  end

  defp mlir_t({"c_ulong", size}) do
    Type.i(size * 8)
  end

  defp mlir_t({"c_long", size}) do
    Type.i(size * 8)
  end

  defp mlir_t({"[*c]" <> _, _}) do
    ~t{!llvm.ptr}
  end

  defp mlir_t({zig_t, size}) do
    if String.ends_with?(zig_t, "enif_environment_t") do
      Type.i(size * 8)
    else
      raise "unsupported Zig type #{zig_t}"
    end
  end

  @doc """
  insert external functions of ENIF into current MLIR block
  """
  def external_functions(ctx, block) do
    mlir ctx: ctx, block: block do
      for {name, arg_types, ret_type} <- MLIR.CAPI.mif_raw_enif_signatures() do
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
    def mlir_t(opts \\ []), do: Type.i64(opts)
  end

  defmodule ERL_NIF_TERM do
    def mlir_t(opts \\ []), do: Type.i64(opts)
  end
end
