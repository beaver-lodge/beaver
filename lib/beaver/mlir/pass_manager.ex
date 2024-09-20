defmodule Beaver.MLIR.PassManager do
  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  alias Beaver.MLIR

  @type print_opts :: [
          before_all: boolean(),
          after_all: boolean(),
          module_scope: boolean(),
          after_only_on_change: boolean(),
          after_only_on_failure: boolean()
        ]
  @spec enable_ir_printing(MLIR.PassManager.t(), print_opts) :: :ok
  def enable_ir_printing(pm, opts \\ []) do
    MLIR.CAPI.mlirPassManagerEnableIRPrinting(
      pm,
      Keyword.get(opts, :before_all, false),
      Keyword.get(opts, :after_all, true),
      Keyword.get(opts, :module_scope, false),
      Keyword.get(opts, :after_only_on_change, false),
      Keyword.get(opts, :after_only_on_failure, false)
    )
  end
end
