defmodule Beaver.MLIR.Pass do
  require Beaver.MLIR.CAPI
  alias Beaver.MLIR
  alias MLIR.CAPI

  use Kinda.ResourceKind,
    fields: [handler: nil],
    forward_module: Beaver.Native

  @callback run(MLIR.Operation.t()) :: :ok | :error

  defmacro __using__(opts) do
    require Beaver.MLIR.CAPI
    alias Beaver.MLIR.Pass.Composer

    quote do
      require Logger
      @behaviour MLIR.Pass

      def create() do
        op_name = Keyword.get(unquote(opts), :on, "builtin.module")

        MLIR.ExternalPass.create(
          __MODULE__,
          op_name
        )
      end

      def delay(%Composer{} = composer_or_op) do
        external_pass = create()
        Composer.add(composer_or_op, external_pass)
      end

      def delay(%Beaver.MLIR.Operation{} = composer_or_op) do
        composer = %Composer{op: composer_or_op, passes: []}
        delay(composer)
      end

      def delay(%Beaver.MLIR.Module{} = composer_or_op) do
        composer = %Composer{op: composer_or_op, passes: []}
        delay(composer)
      end
    end
  end

  def pipeline!(%MLIR.CAPI.MlirOpPassManager{} = pm, pipeline_str) when is_binary(pipeline_str) do
    status = CAPI.mlirParsePassPipeline(pm, MLIR.StringRef.create(pipeline_str))

    if not MLIR.LogicalResult.success?(status) do
      raise "Unexpected failure parsing pipeline: #{pipeline_str}"
    end

    pm
  end

  def pipeline!(%MLIR.CAPI.MlirPassManager{} = pm, pipeline_str) when is_binary(pipeline_str) do
    pm |> CAPI.mlirPassManagerGetAsOpPassManager() |> pipeline!(pipeline_str)
    pm
  end
end
