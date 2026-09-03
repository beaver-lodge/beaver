defmodule Beaver.Triton do
  @moduledoc """
  Registers Triton's core dialects and passes in a Beaver MLIR context.

  Once registered, a context can parse and inspect Triton IR (`tt`, `ttgir`,
  `ttng`, `ttinstrument`, `gluon` dialects), and Triton passes can be driven
  by name through `Beaver.Composer` pipelines.

  Requires a Beaver native build linked against the Triton core prebuilt
  (build with `BEAVER_TRITON_PREBUILT_DIR` set); otherwise the calls raise.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.Triton.PipelinePlan

  defmodule PipelineError do
    @moduledoc "Typed failure raised while executing an auditable Triton pipeline plan."

    defexception [:domain, :pass_id, :pass_index, :message]

    @impl Exception
    def exception(opts) do
      domain = Keyword.fetch!(opts, :domain)
      pass = Keyword.get(opts, :pass)
      index = Keyword.get(opts, :index)
      cause = Keyword.get(opts, :cause)

      %__MODULE__{
        domain: domain,
        pass_id: pass && pass.id,
        pass_index: index,
        message:
          "#{domain}: Triton pipeline pass #{inspect(pass && pass.id)} at index #{inspect(index)} " <>
            "failed: #{Exception.message(cause)}"
      }
    end
  end

  @doc """
  Registers Triton passes and dialects, then loads the dialects on `context`.
  """
  @spec register(MLIR.Context.t()) :: :ok
  def register(%MLIR.Context{} = context) do
    register_passes()

    unless MLIR.CAPI.beaver_raw_triton_register_dialects(context.ref) do
      raise ArgumentError,
            "Beaver was built without Triton support; rebuild the native " <>
              "library with BEAVER_TRITON_PREBUILT_DIR set"
    end

    :ok
  end

  @doc """
  Registers Triton's passes in the global MLIR pass registry.

  Idempotent per process; safe to call multiple times.
  """
  @spec register_passes() :: :ok
  def register_passes do
    unless :persistent_term.get({__MODULE__, :passes_registered}, false) do
      unless MLIR.CAPI.beaver_raw_triton_register_passes() do
        raise ArgumentError,
              "Beaver was built without Triton support; rebuild the native " <>
                "library with BEAVER_TRITON_PREBUILT_DIR set"
      end

      :persistent_term.put({__MODULE__, :passes_registered}, true)
    end

    :ok
  end

  @doc """
  Lowers a TTIR module to LLVM IR through the full NVIDIA Triton pipeline.

  This mirrors Triton's `make_ttgir` + `make_llir` pass chains for a CUDA
  target. Running `convert-triton-gpu-to-llvm` alone is not sufficient for
  real kernels: shared-memory allocation, warp-group allocation, SCF-to-CF,
  warp-specialize lowering, and NVGPU lowering must run first, otherwise
  kernels with layout conversions either fail verification or crash the
  process.

  Requires a context with Triton dialects registered (`register/1`).
  """
  @spec compile_to_llvm(MLIR.Module.t(), keyword()) :: MLIR.Module.t()
  def compile_to_llvm(%MLIR.Module{} = module, opts \\ []) do
    opts
    |> PipelinePlan.build()
    |> execute(module)
  end

  @doc """
  Lowers a Triton module one pass at a time and returns an auditable prefix trace.

  Each pass boundary is verified.  Trace records contain only pass identity,
  rendered options, phase, and a digest of generic MLIR; they do not retain IR.
  """
  @spec compile_to_llvm_with_trace(MLIR.Module.t(), keyword()) ::
          {MLIR.Module.t(), [map()]}
  def compile_to_llvm_with_trace(%MLIR.Module{} = module, opts \\ []) do
    plan = PipelinePlan.build(opts)
    execute_with_trace(plan, module)
  end

  @doc "Returns the pure lowering plan selected by the same options as `compile_to_llvm/2`."
  @spec pipeline_plan(keyword()) :: PipelinePlan.t()
  def pipeline_plan(opts \\ []), do: PipelinePlan.build(opts)

  @doc "Finds the first different pass or prefix digest in two traces."
  @spec first_trace_divergence([map()], [map()]) :: :none | {:ok, map()}
  def first_trace_divergence(left, right) when is_list(left) and is_list(right) do
    max_length = max(length(left), length(right))

    if max_length == 0 do
      :none
    else
      Enum.find_value(0..(max_length - 1), :none, &trace_divergence_at(left, right, &1))
    end
  end

  defp trace_divergence_at(left, right, index) do
    left_record = Enum.at(left, index)
    right_record = Enum.at(right, index)

    if left_record == right_record,
      do: false,
      else: {:ok, %{index: index, left: left_record, right: right_record}}
  end

  defp execute(%PipelinePlan{} = plan, module) do
    Enum.reduce(plan.passes, module, fn pass, acc ->
      Beaver.Composer.append(acc, pass.pipeline)
    end)
    |> Beaver.Composer.run!()
  end

  defp execute_with_trace(%PipelinePlan{} = plan, module) do
    plan_digest = PipelinePlan.digest(plan)

    plan.passes
    |> Enum.with_index()
    |> Enum.reduce({module, []}, fn {pass, index}, {acc, trace} ->
      try do
        lowered = acc |> Beaver.Composer.append(pass.pipeline) |> Beaver.Composer.run!()
        MLIR.verify!(lowered)

        record = %{
          index: index,
          id: pass.id,
          phase: pass.phase,
          pipeline: pass.pipeline,
          plan_digest: plan_digest,
          ir_sha256: ir_digest(lowered)
        }

        {lowered, [record | trace]}
      rescue
        exception ->
          pipeline_error =
            PipelineError.exception(
              domain: failure_domain(exception),
              pass: pass,
              index: index,
              cause: exception
            )

          reraise pipeline_error, __STACKTRACE__
      end
    end)
    |> then(fn {lowered, reversed_trace} -> {lowered, Enum.reverse(reversed_trace)} end)
  end

  defp failure_domain(%RuntimeError{message: message}) do
    if String.contains?(message, "does not refer to a registered pass"),
      do: :pass_registration_missing,
      else: :profile_compile_failed
  end

  defp failure_domain(_exception), do: :prefix_verification_failed

  defp ir_digest(module) do
    module
    |> MLIR.to_string(generic: true)
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end
end
