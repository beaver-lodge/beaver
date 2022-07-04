defmodule Beaver.MLIR.Pass do
  use Beaver
  alias MLIR.CAPI

  defmacro __using__(opts) do
    quote(bind_quoted: [opts: opts]) do
      @behaviour MLIR.Pass
    end
  end

  @callback run(MLIR.CAPI.MlirOperation.t()) :: :ok | :error

  def pipeline!(pm, pipeline_str) when is_binary(pipeline_str) do
    status = CAPI.mlirParsePassPipeline(pm, MLIR.StringRef.create(pipeline_str))

    if not MLIR.LogicalResult.success?(status) do
      raise "Unexpected failure parsing pipeline: #{pipeline_str}"
    end

    pm
  end
end
