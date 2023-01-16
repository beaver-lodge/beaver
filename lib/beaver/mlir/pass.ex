defmodule Beaver.MLIR.Pass do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  require Beaver.MLIR.CAPI
  alias Beaver.MLIR
  alias MLIR.CAPI

  use Kinda.ResourceKind,
    fields: [handler: nil],
    forward_module: Beaver.Native

  @callback run(MLIR.Operation.t()) :: :ok | :error

  defmacro __using__(opts) do
    require Beaver.MLIR.CAPI

    quote do
      @behaviour MLIR.Pass
      Module.register_attribute(__MODULE__, :root_op, persist: true, accumulate: false)
      @root_op Keyword.get(unquote(opts), :on, "builtin.module")
    end
  end

  @doc """
  Parse the string as pass pipeline and add to pass manager
  """
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
