defmodule Beaver.MLIR.PassManager do
  @moduledoc """
  This module defines functions working with MLIR PassManager.
  """
  use Kinda.ResourceKind, forward_module: Beaver.Native
  alias Beaver.MLIR
  import MLIR.CAPI
  require Logger

  @type print_opt ::
          {:before_all, boolean()}
          | {:after_all, boolean()}
          | {:module_scope, boolean()}
          | {:after_only_on_change, boolean()}
          | {:after_only_on_failure, boolean()}
          | {:tree_printing_path, String.t()}
  @type print_opts :: [print_opt()]
  @spec enable_ir_printing(MLIR.PassManager.t(), print_opts()) :: :ok
  def enable_ir_printing(%MLIR.PassManager{} = pm, opts \\ []) do
    MLIR.CAPI.mlirPassManagerEnableIRPrinting(
      pm,
      !!Keyword.get(opts, :before_all, false),
      !!Keyword.get(opts, :after_all, true),
      !!Keyword.get(opts, :module_scope, false),
      !!Keyword.get(opts, :after_only_on_change, false),
      !!Keyword.get(opts, :after_only_on_failure, false),
      MLIR.CAPI.mlirOpPrintingFlagsCreate(),
      MLIR.StringRef.create(Keyword.get(opts, :tree_printing_path, ""))
    )
  end

  def run(%MLIR.PassManager{ref: pm_ref}, op) do
    {status, diagnostics} =
      case beaver_raw_run_pm_on_op_async(pm_ref, MLIR.Operation.from_module(op).ref) do
        :async ->
          dispatch_loop()

        ret ->
          Beaver.Native.check!(ret)
      end

    if MLIR.LogicalResult.success?(status) do
      :ok
    else
      {:error, diagnostics}
    end
  end

  defp dispatch_loop() do
    receive do
      {{:kind, MLIR.LogicalResult, _}, diagnostics} = ret when is_list(diagnostics) ->
        Beaver.Native.check!(ret)

      :ok ->
        :ok

      msg ->
        try do
          :ok = MLIR.Pass.handle_cb(msg)
        rescue
          exception ->
            Logger.error(Exception.format(:error, exception, __STACKTRACE__))
            Logger.flush()
        end

        dispatch_loop()
    end
  end

  def destroy(%__MODULE__{ref: pm_ref}) do
    case beaver_raw_destroy_pm_async(pm_ref) do
      :async ->
        dispatch_loop()

      ret ->
        Beaver.Native.check!(ret)
    end
  end

  defdelegate enable_verifier(pm, enable), to: MLIR.CAPI, as: :mlirPassManagerEnableVerifier
end
