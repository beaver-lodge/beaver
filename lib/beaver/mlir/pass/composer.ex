defmodule Beaver.MLIR.Pass.Composer do
  require Logger

  @moduledoc """
  This module provide functions to compose passes.
  """
  @enforce_keys [:op]
  defstruct passes: [], op: nil
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR

  def new(%__MODULE__{} = composer) do
    composer
  end

  def new(op) do
    %__MODULE__{op: op}
  end

  def append(%op_or_mod{} = composer_or_op, pass)
      when op_or_mod in [MLIR.Module, MLIR.Operation] do
    new(composer_or_op)
    |> append(pass)
  end

  def append(%__MODULE__{passes: passes} = composer, {name, f})
      when is_function(f) do
    %__MODULE__{composer | passes: passes ++ [{name, f}]}
  end

  def append(%__MODULE__{passes: passes} = composer, pass) do
    %__MODULE__{composer | passes: passes ++ [pass]}
  end

  def nested(composer_or_op, op_name, passes) when is_list(passes) do
    new(composer_or_op)
    |> append({:nested, op_name, passes})
  end

  def nested(composer_or_op, op_name, f) when is_function(f, 1) do
    new(composer_or_op)
    |> append({:nested, op_name, f})
  end

  def pipeline(composer_or_op, pipeline_str) when is_binary(pipeline_str) do
    new(composer_or_op)
    |> append(pipeline_str)
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
        {:nested, op_name, f} when (is_binary(op_name) or is_atom(op_name)) and is_function(f, 1) ->
          op_name = get_op_name(op_name)
          npm = mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))
          f.(npm)

        # nest pipeline
        {:nested, op_name, passes}
        when (is_binary(op_name) or is_atom(op_name)) and is_list(passes) ->
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
end
