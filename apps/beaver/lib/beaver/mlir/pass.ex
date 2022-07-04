defmodule Beaver.MLIR.Pass do
  use Beaver
  alias MLIR.CAPI

  @callback run(MLIR.CAPI.MlirOperation.t()) :: :ok | :error

  def run_on_operation(%Beaver.MLIR.CAPI.MlirOperation{} = op, run) when is_function(run, 1) do
    run.(op)
  end

  defmacro __using__(opts) do
    use Beaver
    alias Beaver.MLIR.Pass.Composer

    quote(bind_quoted: [opts: opts]) do
      require Logger
      @behaviour MLIR.Pass

      def handle_invoke(
            :run,
            [
              op,
              %Beaver.MLIR.CAPI.MlirExternalPass{} = pass,
              userData
            ],
            state
          ) do
        with :ok <- Beaver.MLIR.Pass.run_on_operation(op, &run/1) do
          {:return, userData, state}
        else
          :error ->
            Logger.error("fail to run pass #{__MODULE__}")
            :error

          other ->
            Logger.error(
              "must return :ok or :error in run/1 of the pass #{__MODULE__}, get: #{inspect(other)}"
            )

            :error
        end
      end

      def delay(composer_or_op = %Composer{}) do
        # TODO: manage this typeIDAllocator
        type_id_allocator = CAPI.mlirTypeIDAllocatorCreate()

        external_pass =
          %MLIR.CAPI.MlirPass{} = MLIR.ExternalPass.create(__MODULE__, type_id_allocator)

        Composer.add(composer_or_op, external_pass)
      end

      def delay(%Beaver.MLIR.CAPI.MlirOperation{} = composer_or_op) do
        composer = %Composer{op: composer_or_op, passes: []}
        delay(composer)
      end

      def delay(%Beaver.MLIR.CAPI.MlirModule{} = composer_or_op) do
        composer = %Composer{op: composer_or_op, passes: []}
        delay(composer)
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
