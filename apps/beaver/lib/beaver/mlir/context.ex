defmodule Beaver.MLIR.Context do
  alias Beaver.MLIR

  @doc """
  create a MLIR context and register all dialects
  """
  def create(allow_unregistered: allow_unregistered) do
    ctx = MLIR.CAPI.mlirContextCreate()
    MLIR.CAPI.mlirRegisterAllDialects(ctx)
    MLIR.CAPI.mlirContextSetAllowUnregisteredDialects(ctx, allow_unregistered)
    ctx |> Exotic.Value.transmit()
  end

  def create() do
    create(allow_unregistered: false)
  end

  def under_multi_thread(ctx, cb) when is_function(cb, 0) do
    MLIR.CAPI.beaverEnterMultiThreadedExecution(ctx)
    ret = cb.()
    MLIR.CAPI.beaverExitMultiThreadedExecution(ctx)
    ret
  end

  defmacro allow_multi_thread(ctx, do: block) do
    quote bind_quoted: [ctx: ctx, block: block] do
      Beaver.MLIR.Context.under_multi_thread(ctx, fn -> block end)
    end
  end
end
