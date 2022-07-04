defmodule Beaver.MLIR.Pass do
  use Beaver
  alias MLIR.CAPI

  @callback run(MLIR.CAPI.MlirOperation.t()) :: :ok | :error

  defmacro __using__(opts) do
    use Beaver

    quote(bind_quoted: [opts: opts]) do
      @behaviour MLIR.Pass

      def handle_invoke(
            :run = id,
            [
              %Beaver.MLIR.CAPI.MlirOperation{} = op,
              %Beaver.MLIR.CAPI.MlirExternalPass{} = pass,
              userData
            ],
            state
          ) do
        with :ok <- run(op) do
          {:return, userData, state}
        else
          :error -> :error
        end
      end
    end
  end

  def pipeline!(pm, pipeline_str) when is_binary(pipeline_str) do
    status = CAPI.mlirParsePassPipeline(pm, MLIR.StringRef.create(pipeline_str))

    if not MLIR.LogicalResult.success?(status) do
      raise "Unexpected failure parsing pipeline: #{pipeline_str}"
    end

    pm
  end
end
