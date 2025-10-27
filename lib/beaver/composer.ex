defmodule Beaver.Composer do
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR
  require Logger

  @moduledoc """
  This module provide functions to compose and run passes.
  """
  defstruct passes: [], op: nil, ctx: nil
  @type operation :: MLIR.Module.t() | MLIR.Operation.t()
  @type t :: %__MODULE__{passes: list(any()), op: operation}

  def new(ctx: ctx), do: %__MODULE__{ctx: ctx}
  def new(%__MODULE__{} = composer), do: composer

  def new(op), do: %__MODULE__{op: op}

  def append(%op_or_mod{} = composer_or_op, pass)
      when op_or_mod in [MLIR.Module, MLIR.Operation] do
    new(composer_or_op) |> append(pass)
  end

  def append(%__MODULE__{passes: passes} = composer, pass),
    do: %__MODULE__{composer | passes: passes ++ [pass]}

  def nested(composer_or_op, op_name, passes) when is_bitstring(op_name) and is_list(passes) do
    composer_or_op |> append({op_name, passes})
  end

  def nested(composer_or_op, op_name, pass) do
    nested(composer_or_op, op_name, [pass])
  end

  defp op_name_from_persistent_attributes(pass_module) do
    op_name = pass_module.__info__(:attributes)[:root_op] || []
    op_name = op_name |> List.first()
    op_name || "builtin.module"
  end

  @doc false
  def create_pass(%MLIR.Pass{} = pass, _ctx) do
    pass
  end

  def create_pass({argument, op, run}, ctx) when is_function(run) do
    description = "beaver generated pass of #{Function.info(run) |> inspect}"
    MLIR.Pass.create(argument, description, op, run: run, ctx: ctx)
  end

  def create_pass(pass_module, ctx) do
    description = "beaver generated pass of #{pass_module}"
    op_name = op_name_from_persistent_attributes(pass_module)
    name = Atom.to_string(pass_module)

    MLIR.Pass.create(name, description, op_name,
      destruct: &pass_module.destruct/1,
      initialize: &pass_module.initialize/2,
      clone: &pass_module.clone/1,
      run: &pass_module.run/2,
      ctx: ctx
    )
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

  defp add_pass(pm, {op_name, passes}, ctx) when is_binary(op_name) and is_list(passes) do
    npm =
      case pm do
        %MLIR.PassManager{} ->
          mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))

        %MLIR.OpPassManager{} ->
          mlirOpPassManagerGetNestedUnder(pm, MLIR.StringRef.create(op_name))
      end

    for pass <- passes do
      add_pass(npm, pass, ctx)
    end
  end

  defp add_pass(pm, pipeline_str, _ctx) when is_binary(pipeline_str),
    do: add_pipeline(pm, pipeline_str)

  defp add_pass(%MLIR.OpPassManager{} = pm, pass, ctx),
    do: mlirOpPassManagerAddOwnedPass(pm, create_pass(pass, ctx))

  defp add_pass(%MLIR.PassManager{} = pm, pass, ctx),
    do: mlirPassManagerAddOwnedPass(pm, create_pass(pass, ctx))

  defp to_pm(%__MODULE__{passes: passes, op: op, ctx: ctx}) do
    ctx = ctx || MLIR.context(MLIR.Operation.from_module(op))

    pm = mlirPassManagerCreate(ctx)

    for pass <- passes do
      add_pass(pm, pass, ctx)
    end

    pm
  end

  def to_pipeline(composer) do
    pm = composer |> to_pm()
    txt = pm |> MLIR.to_string()
    mlirPassManagerDestroy(pm)
    txt
  end

  @run_default_opts [debug: false, print: false, timing: false, verifier: true]

  @type run_option ::
          {:debug, boolean()}
          | {:print, boolean()}
          | {:timing, boolean()}
          | {:verifier, boolean()}
  @type run_result :: {:ok, any()} | {:error, String.t()}
  @type composer :: __MODULE__.t()
  @spec run!(composer) :: operation
  @spec run!(composer, [run_option]) :: operation

  def run!(
        composer,
        opts \\ @run_default_opts
      ) do
    case run(composer, opts) do
      {:ok, op, []} ->
        op

      {:ok, op, diagnostics} ->
        Logger.warning(
          "Passes completed with diagnostics:\n" <> MLIR.Diagnostic.format(diagnostics)
        )

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
    ctx = MLIR.context(MLIR.Operation.from_module(op))
    pm = init(composer, opts)

    if print do
      mlirContextEnableMultithreading(ctx, false)
      MLIR.PassManager.enable_ir_printing(pm)
    end

    res = MLIR.PassManager.run(pm, op)

    if print do
      mlirContextEnableMultithreading(ctx, true)
    end

    MLIR.PassManager.destroy(pm)

    case res do
      {:ok, diagnostics} -> {:ok, op, diagnostics}
      {:error, diagnostics} -> {:error, diagnostics}
    end
  end

  def init(
        %__MODULE__{} = composer,
        opts \\ @run_default_opts
      ) do
    timing = Keyword.get(opts, :timing)
    debug = Keyword.get(opts, :debug)
    verifier = !!Keyword.get(opts, :verifier)
    pm = to_pm(composer)

    if timing do
      pm |> beaverPassManagerEnableTiming()
    end

    :ok = MLIR.PassManager.enable_verifier(pm, verifier)

    if debug do
      txt = pm |> MLIR.to_string()
      txt = "[pass pipeline] " <> txt
      txt |> Logger.info()
    end

    pm
  end
end
