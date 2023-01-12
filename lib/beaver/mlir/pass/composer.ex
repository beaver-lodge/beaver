defmodule Beaver.MLIR.Pass.Composer do
  require Logger

  @moduledoc """
  This module provide functions to compose passes.
  """
  @enforce_keys [:op]
  defstruct passes: [], op: nil
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR

  def append(%op_or_mod{} = composer_or_op, pass)
      when op_or_mod in [MLIR.Module, MLIR.Operation] do
    %__MODULE__{op: composer_or_op}
    |> append(pass)
  end

  def append(composer = %__MODULE__{passes: passes}, {name, f})
      when is_function(f) do
    %__MODULE__{composer | passes: passes ++ [{name, f}]}
  end

  def append(composer = %__MODULE__{passes: passes}, pass) do
    %__MODULE__{composer | passes: passes ++ [pass]}
  end

  defp get_op_name(op_name) when is_binary(op_name), do: op_name

  # TODO: add keyword arguments
  def run!(
        %__MODULE__{passes: passes, op: %MLIR.Module{} = op},
        opts \\ [dump: false, debug: false, print: false]
      ) do
    print = Keyword.get(opts, :print)
    ctx = MLIR.CAPI.mlirOperationGetContext(MLIR.Operation.from_module(op))
    pm = mlirPassManagerCreate(ctx)

    MLIR.CAPI.mlirPassManagerEnableVerifier(pm, true)

    if print do
      mlirContextEnableMultithreading(ctx, false)
      mlirPassManagerEnableIRPrinting(pm)
      Logger.info("[Beaver] IR printing enabled")
    end

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

    if print do
      mlirContextEnableMultithreading(ctx, true)
    end

    if not MLIR.LogicalResult.success?(status) do
      if Keyword.get(opts, :debug, false) do
        Logger.error("Failed to run pass, start dumping operation and this might crash")
        Logger.info(MLIR.to_string(op))
      end

      raise "Unexpected failure running pass pipeline"
    end

    if Keyword.get(opts, :dump, false), do: MLIR.Operation.dump(op)
    mlirPassManagerDestroy(pm)

    op
  end

  def nested(composer_or_op = %__MODULE__{}, op_name, passes) when is_list(passes) do
    append(composer_or_op, {op_name, passes})
  end

  def nested(composer_or_op, op_name, f) when is_function(f, 1) do
    append(composer_or_op, {op_name, f})
  end

  def nested(composer_or_op, op_name, passes_or_f) do
    composer = %__MODULE__{op: composer_or_op, passes: []}
    nested(composer, op_name, passes_or_f)
  end

  def pipeline(composer_or_op = %__MODULE__{}, pipeline_str) when is_binary(pipeline_str) do
    append(composer_or_op, pipeline_str)
  end

  def pipeline(composer_or_op, pipeline_str) when is_binary(pipeline_str) do
    composer = %__MODULE__{op: composer_or_op, passes: []}
    pipeline(composer, pipeline_str)
  end
end
