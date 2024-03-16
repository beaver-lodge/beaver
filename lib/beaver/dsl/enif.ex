defmodule Beaver.ENIF do
  use Beaver
  require Beaver.MLIR.Dialect.Func
  alias Beaver.MLIR.Dialect.Func
  alias MLIR.Type

  defp wrap_mlir_t({ref, _size}) when is_reference(ref) do
    %MLIR.Type{ref: ref}
  end

  @type opts() :: [ctx: MLIR.Context.t()]
  @type obj() :: :term | :env
  @spec mlir_t(obj(), opts()) :: MLIR.Type.t()
  def mlir_t(obj, opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      fn %MLIR.Context{ref: ref} ->
        ref = MLIR.CAPI.mif_raw_mlir_type_of_enif_obj(ref, obj) |> Beaver.Native.check!()
        %MLIR.Type{ref: ref}
      end
    )
  end

  @doc """
  insert external functions of ENIF into current MLIR block
  """
  def populate_external_functions(ctx, block) do
    mlir ctx: ctx, block: block do
      for {name, arg_types, ret_type} <- signatures(ctx) do
        Func.func _(
                    sym_name: "\"#{name}\"",
                    sym_visibility: MLIR.Attribute.string("private"),
                    function_type: Type.function(arg_types, [ret_type])
                  ) do
          region do
          end
        end
      end
    end
  end

  def signatures(%MLIR.Context{} = ctx) do
    for {name, arg_types, ret_type} <- MLIR.CAPI.mif_raw_enif_signatures(ctx.ref) do
      {name, Enum.map(arg_types, &wrap_mlir_t/1), wrap_mlir_t(ret_type)}
    end
  end

  def signature(%MLIR.Context{} = ctx, name) do
    for {^name, arg_types, ret_type} <- signatures(ctx) do
      {arg_types, ret_type}
    end
    |> List.first()
  end

  defdelegate functions(), to: MLIR.CAPI, as: :mif_raw_enif_functions
end
