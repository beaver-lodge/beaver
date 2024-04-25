defmodule Beaver.MLIR.Pass.Composer do
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR
  require Logger

  @moduledoc """
  This module provide functions to compose passes.
  """
  @enforce_keys [:op]
  defstruct passes: [], op: nil
  @type operation :: MLIR.Module.t() | MLIR.Operation.t()
  @type t :: %__MODULE__{passes: list(any()), op: operation}

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

  defp add_pipeline(%MLIR.OpPassManager{} = pm, pipeline_str)
       when is_binary(pipeline_str) do
    mlirOpPassManagerAddPipeline(
      pm,
      MLIR.StringRef.create(pipeline_str),
      Beaver.Diagnostic.callback(),
      Beaver.Native.OpaquePtr.null()
    )
    |> Beaver.Native.check!()
    |> tap(fn res ->
      if not MLIR.LogicalResult.success?(res) do
        raise "Unexpected failure parsing pipeline: #{pipeline_str}"
      end
    end)

    pm
  end

  defp add_pipeline(%MLIR.PassManager{} = pm, pipeline_str)
       when is_binary(pipeline_str) do
    pm |> mlirPassManagerGetAsOpPassManager() |> add_pipeline(pipeline_str)
  end

  # nested pm

  defp add_pass(pm, {:nested, op_name, passes}) when is_binary(op_name) and is_list(passes) do
    npm =
      case pm do
        %MLIR.PassManager{} ->
          mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))

        %MLIR.OpPassManager{} ->
          mlirOpPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))
      end

    for pass <- passes do
      add_pass(npm, pass)
    end
  end

  defp add_pass(pm, pipeline_str) when is_binary(pipeline_str),
    do: add_pipeline(pm, pipeline_str)

  defp add_pass(%MLIR.OpPassManager{} = pm, pass),
    do: mlirOpPassManagerAddOwnedPass(pm, create_pass(pass))

  defp add_pass(%MLIR.PassManager{} = pm, pass),
    do: mlirPassManagerAddOwnedPass(pm, create_pass(pass))

  defp to_pm(%__MODULE__{passes: passes, op: op}) do
    ctx = mlirOperationGetContext(MLIR.Operation.from_module(op))

    pm = mlirPassManagerCreate(ctx)

    for pass <- passes do
      add_pass(pm, pass)
    end

    pm
  end

  def to_pipeline(composer) do
    pm = composer |> to_pm()
    txt = pm |> MLIR.to_string()
    mlirPassManagerDestroy(pm)
    txt
  end

  @run_default_opts [debug: false, print: false, timing: false]

  @type run_option :: {:debug, boolean()} | {:print, boolean()} | {:timing, boolean()}
  @type run_result :: {:ok, any()} | {:error, String.t()}
  @type composer :: __MODULE__.t()
  @spec run!(composer) :: operation
  @spec run!(composer, [run_option]) :: operation

  def run!(
        composer,
        opts \\ @run_default_opts
      ) do
    case run(composer, opts) do
      {:ok, op} ->
        op

      {:error, msg} ->
        raise msg
    end
  end

  @spec run(composer) :: run_result
  @spec run(composer, [run_option]) :: run_result

  def run(
        %__MODULE__{op: op} = composer,
        opts \\ @run_default_opts
      ) do
    print = Keyword.get(opts, :print)
    timing = Keyword.get(opts, :timing)
    debug = Keyword.get(opts, :debug)
    ctx = mlirOperationGetContext(MLIR.Operation.from_module(op))

    pm = to_pm(composer)

    if timing do
      pm |> beaverPassManagerEnableTiming()
    end

    mlirPassManagerEnableVerifier(pm, true)

    if print do
      mlirContextEnableMultithreading(ctx, false)
      mlirPassManagerEnableIRPrinting(pm)
      Logger.info("[Beaver] IR printing enabled")
    end

    if debug do
      txt = pm |> MLIR.to_string()
      txt = "[pass pipeline] " <> txt
      txt |> Logger.info()
    end

    status = mlirPassManagerRunOnOp(pm, MLIR.Operation.from_module(op))

    if print do
      mlirContextEnableMultithreading(ctx, true)
    end

    mlirPassManagerDestroy(pm)

    if MLIR.LogicalResult.success?(status) do
      {:ok, op}
    else
      {:error, "Unexpected failure running passes"}
    end
  end
end
