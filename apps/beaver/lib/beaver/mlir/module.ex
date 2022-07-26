defmodule Beaver.MLIR.Module do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  def create(str, opts \\ [])

  def create(str, opts) when is_binary(str) and is_list(opts) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    create(ctx, str)
  end

  def create(context, str) when is_binary(str) do
    CAPI.mlirModuleCreateParse(context, MLIR.StringRef.create(str))
  end

  def create!(context, str) when is_binary(str) do
    module = create(context, str)
    verify!(module)
    module
  end

  use Fizz.ResourceKind,
    root_module: CAPI,
    forward_module: Beaver.Native

  def is_null(module) do
    CAPI.beaverModuleIsNull(module) |> Beaver.Native.to_term()
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
    CAPI.mlirModuleDestroy(module)
  end
end
