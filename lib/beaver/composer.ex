defmodule Beaver.Composer do
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR
  require Logger

  @moduledoc """
  This module provide functions to compose and run passes.
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

  defp op_name_from_persistent_attributes(pass_module) do
    op_name = pass_module.__info__(:attributes)[:root_op] || []
    op_name = op_name |> List.first()
    op_name || "builtin.module"
  end

  # Create an external pass.
  defp do_create_pass(pid, argument, description, op, run) do
    argument_ref = MLIR.StringRef.create(argument).ref

    MLIR.CAPI.beaver_raw_create_mlir_pass(
      argument_ref,
      argument_ref,
      MLIR.StringRef.create(description).ref,
      MLIR.StringRef.create(op).ref,
      pid,
      run
    )
    |> Beaver.Native.check!()
  end

  defp create_pass(argument, desc, op, run) do
    do_create_pass(self(), argument, desc, op, run)
  end

  def create_pass(%MLIR.Pass{} = pass) do
    pass
  end

  def create_pass({argument, op, run}) when is_function(run) do
    description = "beaver generated pass of #{Function.info(run) |> inspect}"
    create_pass(argument, description, op, run)
  end

  def create_pass(pass_module) do
    description = "beaver generated pass of #{pass_module}"
    op_name = op_name_from_persistent_attributes(pass_module)
    name = Atom.to_string(pass_module)
    create_pass(name, description, op_name, &pass_module.run/1)
  end

  defp add_pipeline(%MLIR.OpPassManager{} = pm, pipeline_str)
       when is_binary(pipeline_str) do
    {res, err} =
      Beaver.Printer.run(
        &mlirOpPassManagerAddPipeline(
          pm,
          MLIR.StringRef.create(pipeline_str),
          &1,
          &2
        )
      )

    if not MLIR.LogicalResult.success?(res) do
      raise "Unexpected failure parsing pipeline: #{pipeline_str}, #{err}"
    end

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
    ctx = MLIR.context(MLIR.Operation.from_module(op))

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

  defp dispatch_pass_action() do
    receive do
      {:run, op_ref, token_ref, run} ->
        spawn_link(fn ->
          try do
            run.(%MLIR.Operation{ref: op_ref})
            MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref)
          rescue
            exception ->
              MLIR.CAPI.beaver_raw_logical_mutex_token_signal_failure(token_ref)
              Logger.error(Exception.format(:error, exception, __STACKTRACE__))
          end
        end)

        dispatch_pass_action()

      {{:kind, MLIR.LogicalResult, _}, diagnostics} = ret when is_list(diagnostics) ->
        Beaver.Native.check!(ret)

      other ->
        Logger.error("Unexpected message: #{inspect(other)}")
        dispatch_pass_action()
    end
  end

  def run!(
        composer,
        opts \\ @run_default_opts
      ) do
    case run(composer, opts) do
      {:ok, op} ->
        op

      {:error, diagnostics} ->
        raise ArgumentError,
              MLIR.Diagnostic.format(diagnostics, "Unexpected failure running passes")
    end
  end

  @spec run(composer) :: run_result
  @spec run(composer, [run_option]) :: run_result

  @doc """
  Run the passes on the operation.

  > #### Must be a multi-threaded context if an Elixir pass is in the pipeline {: .info}
  >
  > MLIR context's thread pool is used to run the CAPI. If an Elixir pass is in the pipeline, the context must be multi-threaded otherwise there can be a deadlock. Also note that it can be more expensive than a C/C++ implementation due to the overhead of the thread pool.
  """
  def run(
        %__MODULE__{op: op} = composer,
        opts \\ @run_default_opts
      ) do
    print = Keyword.get(opts, :print)
    timing = Keyword.get(opts, :timing)
    debug = Keyword.get(opts, :debug)
    ctx = MLIR.context(MLIR.Operation.from_module(op))

    pm = to_pm(composer)

    if timing do
      pm |> beaverPassManagerEnableTiming()
    end

    mlirPassManagerEnableVerifier(pm, true)

    if print do
      mlirContextEnableMultithreading(ctx, false)
      MLIR.PassManager.enable_ir_printing(pm)
    end

    if debug do
      txt = pm |> MLIR.to_string()
      txt = "[pass pipeline] " <> txt
      txt |> Logger.info()
    end

    {status, diagnostics} = run_pm_async(pm, op)

    if print do
      mlirContextEnableMultithreading(ctx, true)
    end

    mlirPassManagerDestroy(pm)

    if MLIR.LogicalResult.success?(status) do
      {:ok, op}
    else
      {:error, diagnostics}
    end
  end

  @doc false
  def run_pm_async(%MLIR.PassManager{ref: pm_ref}, op) do
    case beaver_raw_run_pm_on_op_async(pm_ref, MLIR.Operation.from_module(op).ref) do
      :ok ->
        dispatch_pass_action()

      ret ->
        Beaver.Native.check!(ret)
    end
  end
end
