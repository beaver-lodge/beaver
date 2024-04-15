defmodule Beaver.ENIF do
  @moduledoc """
  This module provides functions to work with Erlang's [erl_nif](https://www.erlang.org/doc/man/erl_nif.html) APIs in MLIR.
  """
  use Beaver
  require Beaver.MLIR.Dialect.Func
  alias Beaver.MLIR.Dialect.Func
  alias MLIR.Type

  defp wrap_mlir_t({ref, _size}) when is_reference(ref) do
    %MLIR.Type{ref: ref}
  end

  @doc """
  insert external functions of ENIF into current MLIR block
  """
  def populate_external_functions(ctx, block) do
    mlir ctx: ctx, block: block do
      for {name, arg_types, ret_types} <- signatures(ctx) do
        Func.func _(
                    sym_name: "\"#{name}\"",
                    sym_visibility: MLIR.Attribute.string("private"),
                    function_type: Type.function(arg_types, ret_types)
                  ) do
          region do
          end
        end
      end
    end
  end

  def signatures(%MLIR.Context{} = ctx) do
    signatures = MLIR.CAPI.mif_raw_enif_signatures(ctx.ref) |> Beaver.Native.check!()

    for {name, arg_types, ret_types} <- signatures do
      {name, Enum.map(arg_types, &wrap_mlir_t/1), Enum.map(ret_types, &wrap_mlir_t/1)}
    end
  end

  def signature(%MLIR.Context{} = ctx, name) do
    for {^name, arg_types, ret_types} <- signatures(ctx) do
      {arg_types, ret_types}
    end
    |> List.first()
  end

  defdelegate functions(), to: MLIR.CAPI, as: :mif_raw_enif_functions
end
