defmodule Beaver.MLIR.Pass.Composer do
  @moduledoc """
  This module provide functions to compose passes.
  """
  @enforce_keys [:passes, :op]
  defstruct passes: [], op: nil
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR

  def add(composer = %__MODULE__{passes: passes}, pass) do
    %__MODULE__{composer | passes: passes ++ [pass]}
  end

  # TODO: add keyword arguments
  def run!(%__MODULE__{passes: passes, op: op}) do
    ctx = MLIR.Managed.Context.get()
    pm = mlirPassManagerCreate(ctx)

    for pass <- passes do
      case pass do
        {op_name, passes} when is_binary(op_name) ->
          npm = mlirOpPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))

          for pass <- passes do
            mlirPassManagerAddOwnedPass(npm, pass)
          end

        pipeline_str when is_binary(pipeline_str) ->
          MLIR.Pass.pipeline!(pm, pipeline_str)

        _ ->
          mlirPassManagerAddOwnedPass(pm, pass)
      end
    end

    status = mlirPassManagerRun(pm, op)

    if not MLIR.LogicalResult.success?(status) do
      raise "Unexpected failure running pass pipeline"
    end

    mlirPassManagerDestroy(pm)
    op
  end

  def nested(composer_or_op = %__MODULE__{}, op_name, passes) when is_list(passes) do
    __MODULE__.add(composer_or_op, {op_name, passes})
  end

  def nested(composer_or_op, op_name, pass) do
    composer = %__MODULE__{op: composer_or_op, passes: []}
    nested(composer, op_name, pass)
  end

  def pipeline(composer_or_op = %__MODULE__{}, pipeline_str) when is_binary(pipeline_str) do
    __MODULE__.add(composer_or_op, pipeline_str)
  end

  def pipeline(composer_or_op, pipeline_str) when is_binary(pipeline_str) do
    composer = %__MODULE__{op: composer_or_op, passes: []}
    pipeline(composer, pipeline_str)
  end
end
