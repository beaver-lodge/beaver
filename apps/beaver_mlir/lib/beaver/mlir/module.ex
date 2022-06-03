defmodule Beaver.MLIR.Module do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI.IR
  import Beaver.MLIR.CAPI

  def create(str, opts \\ [])

  def create(str, opts) when is_binary(str) and is_list(opts) do
    ctx =
      with ctx = %Exotic.Value{} <- opts[:ctx] do
        ctx
      else
        nil ->
          MLIR.Managed.Context.get()
      end

    create(ctx, str)
  end

  def create(context, str) when is_binary(str) do
    IR.mlirModuleCreateParse(context, IR.string_ref(str))
  end

  def create!(context, str) when is_binary(str) do
    module = create(context, str)
    verify!(module)
    module
  end

  def is_null(module) do
    module
    |> Exotic.Value.fetch(MLIR.CAPI.MlirModule, :ptr)
    |> Exotic.Value.extract() == 0
  end

  defp not_null!(module) do
    if is_null(module) do
      raise "module is null"
    end
  end

  def verify!(module) do
    not_null!(module)
    MLIR.Operation.verify!(module)
  end

  def destroy(module) do
    mlirModuleDestroy(module)
  end
end
