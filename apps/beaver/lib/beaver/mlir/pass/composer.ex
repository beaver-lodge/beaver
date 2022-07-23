defmodule Beaver.MLIR.Pass.Composer do
  require Logger

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

  defp get_op_name(op_name) when is_binary(op_name), do: op_name

  defp get_op_name(op_module) when is_atom(op_module),
    do: Beaver.DSL.Op.Prototype.op_name!(op_module)

  # TODO: add keyword arguments
  def run!(
        %__MODULE__{passes: passes, op: %MLIR.CAPI.MlirModule{} = op},
        opts \\ [dump: false, dump_if_fail: false]
      ) do
    ctx = MLIR.Managed.Context.get()
    pm = mlirPassManagerCreate(ctx)

    for pass <- passes do
      case pass do
        # nested pm
        {op_name, f} when (is_binary(op_name) or is_atom(op_name)) and is_function(f, 1) ->
          op_name = get_op_name(op_name)
          npm = mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))
          f.(npm)

        # nest pipeline
        {op_name, passes} when (is_binary(op_name) or is_atom(op_name)) and is_list(passes) ->
          op_name = get_op_name(op_name)
          npm = mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))

          for pass <- passes do
            case pass do
              p when is_binary(p) ->
                MLIR.Pass.pipeline!(npm, p)

              _ ->
                mlirOpPassManagerAddOwnedPass(npm, pass)
            end
          end

        pipeline_str when is_binary(pipeline_str) ->
          MLIR.Pass.pipeline!(pm, pipeline_str)

        _ ->
          mlirPassManagerAddOwnedPass(pm, pass)
      end
    end

    status = mlirPassManagerRun(pm, op)

    if not MLIR.LogicalResult.success?(status) do
      if Keyword.get(opts, :dump_if_fail, false) do
        Logger.error("Failed to run pass, start dumping operation and this might crash")
        MLIR.Operation.dump(op)
      end

      raise "Unexpected failure running pass pipeline"
    end

    if Keyword.get(opts, :dump, false), do: MLIR.Operation.dump(op)
    mlirPassManagerDestroy(pm)
    op
  end

  def nested(composer_or_op = %__MODULE__{}, op_name, passes) when is_list(passes) do
    __MODULE__.add(composer_or_op, {op_name, passes})
  end

  def nested(composer_or_op, op_name, f) when is_function(f, 1) do
    __MODULE__.add(composer_or_op, {op_name, f})
  end

  def nested(composer_or_op, op_name, passes_or_f) do
    composer = %__MODULE__{op: composer_or_op, passes: []}
    nested(composer, op_name, passes_or_f)
  end

  def pipeline(composer_or_op = %__MODULE__{}, pipeline_str) when is_binary(pipeline_str) do
    __MODULE__.add(composer_or_op, pipeline_str)
  end

  def pipeline(composer_or_op, pipeline_str) when is_binary(pipeline_str) do
    composer = %__MODULE__{op: composer_or_op, passes: []}
    pipeline(composer, pipeline_str)
  end
end
