defmodule Beaver.MLIR.Pass do
  use Beaver
  alias MLIR.CAPI

  @callback run(MLIR.CAPI.MlirOperation.t()) :: :ok | :error

  def get_op_name(opts) do
    op_module = Keyword.get(opts, :on, MLIR.Dialect.Func.Func)
    Beaver.DSL.Op.Prototype.op_name!(op_module)
  end

  defmacro __using__(opts) do
    use Beaver
    alias Beaver.MLIR.Pass.Composer

    quote do
      require Logger
      @behaviour MLIR.Pass

      def create() do
        MLIR.ExternalPass.create(
          __MODULE__,
          Beaver.MLIR.Pass.get_op_name(unquote(opts))
        )
      end

      def delay(composer_or_op = %Composer{}) do
        external_pass = create()
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
