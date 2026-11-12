alias Beaver.MLIR.{Operation, StringRef}

defmodule Beaver.Composer do
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR
  require Logger

  @moduledoc "This module provide functions to compose and run passes."
  defstruct passes: [], op: nil, ctx: nil
  @type operation :: MLIR.Module.t() | Operation.t()
  @type t :: %__MODULE__{passes: list(any()), op: operation}

  def new(ctx: ctx), do: %__MODULE__{ctx: ctx}
  def new(%__MODULE__{} = composer), do: composer

  def new(op), do: %__MODULE__{op: op}

  def append(%op_or_mod{} = composer_or_op, pass) when op_or_mod in [MLIR.Module, Operation],
    do: new(composer_or_op) |> append(pass)

  def append(%__MODULE__{passes: passes} = composer, pass), do: %__MODULE__{composer | passes: passes ++ [pass]}

  def nested(composer_or_op, op_name, passes) when is_bitstring(op_name) and is_list(passes),
    do: composer_or_op |> append({op_name, passes})

  def nested(composer_or_op, op_name, pass), do: nested(composer_or_op, op_name, [pass])

  defp op_name_from_persistent_attributes(pass_module) do
    op_name = pass_module.__info__(:attributes)[:root_op] || []
    op_name = op_name |> List.first()
    op_name || "builtin.module"
  end

  @doc false
  def create_pass(%MLIR.Pass{} = pass, _ctx), do: pass

  def create_pass(%MLIR.Transform.FixedPoint{} = fixed_point, ctx) do
    inner_pipeline = beaverOpPassManagerCreate()

    try do
      Enum.each(fixed_point.pipeline, &add_pass(inner_pipeline, &1, ctx))

      beaverCreateCompositeFixedPointPass(
        StringRef.create(fixed_point.name),
        inner_pipeline,
        fixed_point.max_iterations,
        encode_convergence_failure_action(fixed_point.on_convergence_failure)
      )
    after
      beaverOpPassManagerDestroy(inner_pipeline)
    end
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

  defp add_pipeline(%MLIR.OpPassManager{} = pm, pipeline_str) when is_binary(pipeline_str) do
    {res, err} = Beaver.Printer.run(&mlirOpPassManagerAddPipeline(pm, StringRef.create(pipeline_str), &1, &2))

    if not MLIR.LogicalResult.success?(res), do: raise("Unexpected failure parsing pipeline: #{pipeline_str}, #{err}")
    pm
  end

  defp add_pipeline(%MLIR.PassManager{} = pm, pipeline_str) when is_binary(pipeline_str),
    do: pm |> mlirPassManagerGetAsOpPassManager() |> add_pipeline(pipeline_str)

  # nested pm

  defp add_pass(pm, {op_name, passes}, ctx) when is_binary(op_name) and is_list(passes) do
    npm =
      case pm do
        %MLIR.PassManager{} -> mlirPassManagerGetNestedUnder(pm, StringRef.create(op_name))
        %MLIR.OpPassManager{} -> mlirOpPassManagerGetNestedUnder(pm, StringRef.create(op_name))
      end

    for pass <- passes, do: add_pass(npm, pass, ctx)
  end

  defp add_pass(pm, pipeline_str, _ctx) when is_binary(pipeline_str), do: add_pipeline(pm, pipeline_str)

  defp add_pass(%MLIR.OpPassManager{} = pm, pass, ctx), do: mlirOpPassManagerAddOwnedPass(pm, create_pass(pass, ctx))

  defp add_pass(%MLIR.PassManager{} = pm, pass, ctx), do: mlirPassManagerAddOwnedPass(pm, create_pass(pass, ctx))

  defp encode_convergence_failure_action(:warn), do: 0
  defp encode_convergence_failure_action(:error), do: 1
  defp encode_convergence_failure_action(:silent), do: 2

  defp to_pm(%__MODULE__{passes: passes, op: op, ctx: ctx}, anchor \\ "builtin.module") do
    ctx = ctx || MLIR.context(Operation.from_module(op))
    pm = mlirPassManagerCreateOnOperation(ctx, StringRef.create(anchor))

    try do
      for pass <- passes, do: add_pass(pm, pass, ctx)
      pm
    catch
      kind, reason ->
        MLIR.PassManager.destroy(pm)
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
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
          | {:verifier, boolean() | :pipeline}
          | {:execution, :native | :beam}
          | {:verify_execution, :worker | :dirty | :auto}
          | {:profile, Beaver.Execution.Profile.t()}
  @type run_result :: {:ok, any()} | {:error, String.t()}
  @type composer :: __MODULE__.t()
  @spec run!(composer) :: operation
  @spec run!(composer, [run_option]) :: operation

  def run!(composer, opts \\ @run_default_opts), do: unwrap_result!(run(composer, opts))

  @doc false
  def unwrap_result!({:ok, op, []}), do: op

  def unwrap_result!({:ok, op, diagnostics}) do
    Logger.warning("Passes completed with diagnostics:\n" <> MLIR.Diagnostic.format(diagnostics))
    op
  end

  def unwrap_result!({:error, diagnostics}),
    do: raise(ArgumentError, MLIR.Diagnostic.format(diagnostics, "Unexpected failure running passes"))

  @spec run(composer) :: run_result
  @spec run(composer, [run_option]) :: run_result

  @doc """
  Run the passes on the operation.

  > #### Elixir callback safety {: .info}
  >
  > Native pipelines use a separate callback executor outside BEAM schedulers.
  > With `execution: :beam`, Elixir passes run directly in the calling process;
  > native stages still use MLIR's pass manager. Nested pipelines run in IR order
  > with private pass state. Analyses are not preserved across BEAM stages.
  > Per-pass verification uses `verify_execution: :auto`; `:worker` and `:dirty`
  > select an explicit strategy without changing verification frequency.
  > BEAM mode also accepts `verifier: :pipeline` to verify the final IR once,
  > avoiding a native verification round trip after every Elixir pass.
  """
  def run(composer, opts \\ @run_default_opts) do
    Beaver.MLIR.AnalysisSession.track(opts[:analysis_session], fn ->
      execute(composer, Keyword.delete(opts, :analysis_session))
    end)
  end

  defp execute(composer, opts) do
    case Keyword.get(opts, :execution, :native) do
      :native ->
        if opts[:verifier] == :pipeline, do: raise(ArgumentError, "verifier: :pipeline requires execution: :beam")

        run_native(composer, opts)

      :beam ->
        __MODULE__.Beam.run(composer, opts)

      other ->
        raise ArgumentError, "unsupported pass execution mode: #{inspect(other)}"
    end
  end

  @doc "Resolves a reusable BEAM pipeline template without constructing mutable pass state."
  def prepare(%__MODULE__{} = composer, opts \\ []) do
    reusable!(composer.passes)
    __MODULE__.Beam.prepare(composer, opts)
  end

  defp reusable!(passes) do
    Enum.each(passes, fn
      %MLIR.Pass{} -> raise ArgumentError, "prepared pipelines require pass declarations, not owned pass handles"
      %MLIR.Transform.FixedPoint{pipeline: passes} -> reusable!(passes)
      {_root, passes} when is_list(passes) -> reusable!(passes)
      _ -> :ok
    end)
  end

  @doc "Runs the pipeline and returns its result with a unified execution cost receipt."
  def profile(composer, opts \\ []),
    do: Beaver.Execution.Profile.capture(fn profile -> run(composer, Keyword.put(opts, :profile, profile)) end)

  defp run_native(%__MODULE__{op: op} = composer, opts) do
    print = Keyword.get(opts, :print)
    pm = Beaver.Execution.Profile.measure(opts[:profile], :prepare, fn -> init(composer, opts) end)
    # Local-scope printing does not require changing the shared context's
    # threading configuration (which also changes its locking mode).
    if print, do: MLIR.PassManager.enable_ir_printing(pm)

    res =
      try do
        MLIR.PassManager.run(pm, op, Keyword.take(opts, [:profile]))
      after
        Beaver.Execution.Profile.measure(opts[:profile], :cleanup, fn -> MLIR.PassManager.destroy(pm) end)
      end

    case res do
      {:ok, diagnostics} -> {:ok, op, diagnostics}
      {:error, diagnostics} -> {:error, diagnostics}
    end
  end

  def init(composer, opts \\ @run_default_opts), do: init(composer, opts, "builtin.module")

  @doc false
  def init(%__MODULE__{} = composer, opts, anchor) do
    timing = Keyword.get(opts, :timing)
    debug = Keyword.get(opts, :debug)
    verifier = Keyword.get(opts, :verifier, true)
    pm = to_pm(composer, anchor)
    if timing, do: pm |> mlirPassManagerEnableTiming()
    :ok = MLIR.PassManager.enable_verifier(pm, verifier)

    if debug do
      txt = pm |> MLIR.to_string()
      txt = "[pass pipeline] " <> txt
      txt |> Logger.info()
    end

    pm
  end
end
