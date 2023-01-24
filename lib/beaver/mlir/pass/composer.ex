defmodule Beaver.MLIR.Pass.Composer do
  require Logger

  @moduledoc """
  This module provide functions to compose passes.
  """
  @enforce_keys [:op]
  defstruct passes: [], op: nil
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR

  def new(%__MODULE__{} = composer), do: composer

  def new(op), do: %__MODULE__{op: op}

  def append(%op_or_mod{} = composer_or_op, pass)
      when op_or_mod in [MLIR.Module, MLIR.Operation] do
    new(composer_or_op) |> append(pass)
  end

  def append(%__MODULE__{passes: passes} = composer, pass),
    do: %__MODULE__{composer | passes: passes ++ [pass]}

  def nested(composer_or_op, op_name, passes) when is_list(passes) do
    composer_or_op |> append({:nested, op_name, passes})
  end

  def nested(composer_or_op, op_name, pass) do
    nested(composer_or_op, op_name, [pass])
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

  defp add_pipeline(%MLIR.CAPI.MlirOpPassManager{} = pm, pipeline_str)
       when is_binary(pipeline_str) do
    ref =
      MLIR.CAPI.beaver_raw_parse_pass_pipeline(pm.ref, MLIR.StringRef.create(pipeline_str).ref)
      |> Beaver.Native.check!()

    status = %MLIR.LogicalResult{ref: ref}

    if not MLIR.LogicalResult.success?(status) do
      raise "Unexpected failure parsing pipeline: #{pipeline_str}"
    end

    pm
  end

  defp add_pipeline(%MLIR.CAPI.MlirPassManager{} = pm, pipeline_str)
       when is_binary(pipeline_str) do
    pm |> MLIR.CAPI.mlirPassManagerGetAsOpPassManager() |> add_pipeline(pipeline_str)
  end

  # nested pm

  defp add_pass(pm, {:nested, op_name, passes}) when is_binary(op_name) and is_list(passes) do
    for pass <- passes do
      npm =
        case pm do
          %MLIR.CAPI.MlirPassManager{} ->
            mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))

          %MLIR.CAPI.MlirOpPassManager{} ->
            mlirOpPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))
        end

      case pass do
        {:nested, op_name, passes} ->
          add_pass(npm, {:nested, op_name, passes})

        _ ->
          add_pass(npm, pass)
      end
    end
  end

  defp add_pass(pm, pipeline_str) when is_binary(pipeline_str),
    do: add_pipeline(pm, pipeline_str)

  defp add_pass(%MLIR.CAPI.MlirOpPassManager{} = pm, pass),
    do: mlirOpPassManagerAddOwnedPass(pm, create_pass(pass))

  defp add_pass(%MLIR.CAPI.MlirPassManager{} = pm, pass),
    do: mlirPassManagerAddOwnedPass(pm, create_pass(pass))

  def run!(
        %__MODULE__{passes: passes, op: %MLIR.Module{} = op},
        opts \\ [dump: false, debug: false, print: false, timing: false]
      ) do
    print = Keyword.get(opts, :print)
    timing = Keyword.get(opts, :timing)
    debug = Keyword.get(opts, :debug)
    ctx = MLIR.CAPI.mlirOperationGetContext(MLIR.Operation.from_module(op))
    pm = mlirPassManagerCreate(ctx)

    if timing do
      pm |> beaverPassManagerEnableTiming()
    end

    MLIR.CAPI.mlirPassManagerEnableVerifier(pm, true)

    if print do
      mlirContextEnableMultithreading(ctx, false)
      mlirPassManagerEnableIRPrinting(pm)
      Logger.info("[Beaver] IR printing enabled")
    end

    for pass <- passes do
      add_pass(pm, pass)
    end

    if debug do
      txt = pm |> MLIR.to_string()
      txt = "[pass pipeline] " <> txt
      txt |> Logger.info()
    end

    status = mlirPassManagerRun(pm, op)

    if print do
      mlirContextEnableMultithreading(ctx, true)
    end

    mlirPassManagerDestroy(pm)

    if not MLIR.LogicalResult.success?(status) do
      raise "Unexpected failure running passes"
    end

    if Keyword.get(opts, :dump, false), do: MLIR.Operation.dump(op)

    op
  end
end
