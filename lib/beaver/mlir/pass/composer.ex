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

  defp create_pass(pass_module) when is_atom(pass_module) do
    MLIR.ExternalPass.create(pass_module)
  end

  defp create_pass(%MLIR.Pass{} = pass) do
    pass
  end

  defp create_pass({argument, op, run}) when is_bitstring(op) and is_function(run) do
    MLIR.ExternalPass.create({argument, op, run})
  end

  # nested pm
  defp add_pass(pm, {:nested, op_name, f})
       when (is_binary(op_name) or is_atom(op_name)) and is_function(f, 1) do
    npm = mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))
    f.(npm)
  end

  # nest pipeline
  defp add_pass(pm, {:nested, op_name, passes})
       when (is_binary(op_name) or is_atom(op_name)) and is_list(passes) do
    npm = mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))
    for pass <- passes, do: add_pass(npm, pass)
  end

  defp add_pass(pm, pipeline_str) when is_binary(pipeline_str),
    do: MLIR.Pass.pipeline!(pm, pipeline_str)

  defp add_pass(%MLIR.CAPI.MlirOpPassManager{} = pm, pass),
    do: mlirOpPassManagerAddOwnedPass(pm, create_pass(pass))

  defp add_pass(pm, pass), do: mlirPassManagerAddOwnedPass(pm, create_pass(pass))

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

    for pass <- passes, do: add_pass(pm, pass)

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
