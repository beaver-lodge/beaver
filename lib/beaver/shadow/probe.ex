defmodule Beaver.Shadow.Probe do
  @moduledoc """
  Runs a corpus fixture through the Triton lowering pipeline in a child BEAM
  and records the outcome as a deterministic failure receipt.

  Some lowering passes in the pinned Triton prebuilt crash the process
  (segfault) instead of raising, so the probe runs `evaluate/1` in a fresh
  `elixir` subprocess reusing the caller's compiled code path. A crash is
  reported with the offending pass, taken from a marker file the child
  updates before running each pass; a clean Elixir error is reported with
  its message; a successful lowering reports whether `llvm.func` was
  produced. The same evaluation is available interactively through
  `mix shadow.probe <fixture>`.
  """

  alias Beaver.MLIR
  alias Beaver.Shadow.Corpus
  alias Beaver.Shadow.Tuning

  @child_prefix "SHADOW_PROBE "
  @marker_env "SHADOW_PROBE_MARKER"

  defmodule Result do
    @moduledoc "One probe outcome for one corpus fixture."
    @enforce_keys [:fixture, :status]
    defstruct [:fixture, :status, :detail, :exit_code]

    @type status() :: :ok | :error | :crash
    @type t() :: %__MODULE__{
            fixture: atom(),
            status: status(),
            detail: term(),
            exit_code: integer() | nil
          }
  end

  @doc """
  Probes `fixture_name` in a child BEAM process.

  Returns a `Beaver.Shadow.Probe.Result`; the caller's process never sees the
  child's native crash.
  """
  @spec run(atom(), keyword()) :: Result.t()
  def run(fixture_name, opts \\ []) when is_atom(fixture_name) do
    fixture = Corpus.fixture(fixture_name)
    marker = marker_path()

    {output, exit_code} =
      System.cmd(elixir_executable(), child_args(fixture_name, opts),
        stderr_to_stdout: true,
        cd: System.tmp_dir!(),
        env: [{@marker_env, marker}]
      )

    result =
      case parse_child_output(output) do
        %{"status" => "ok", "llvm_func" => llvm_func} ->
          %Result{fixture: fixture.name, status: :ok, detail: %{llvm_func: llvm_func}}

        %{"status" => "error", "message" => message} ->
          %Result{fixture: fixture.name, status: :error, detail: message}

        nil ->
          %Result{
            fixture: fixture.name,
            status: :crash,
            exit_code: exit_code,
            detail: %{last_pass: last_pass(marker)}
          }
      end

    File.rm(marker)
    result
  end

  @doc """
  Probes every `Beaver.Shadow.Tuning.Config` in a fresh child BEAM and
  returns one result per config, so a config space stays isolated and
  attributed even when some configs crash the native passes.
  """
  @spec probe_many(atom(), [Tuning.Config.t()]) :: [
          %{config: Tuning.Config.t(), result: Result.t()}
        ]
  def probe_many(fixture_name, configs) when is_atom(fixture_name) and is_list(configs) do
    Enum.map(configs, fn config ->
      %{config: config, result: run(fixture_name, num_warps: config.num_warps)}
    end)
  end

  defp elixir_executable do
    System.find_executable("elixir") ||
      raise ArgumentError, "elixir executable not found for the probe subprocess"
  end

  defp child_args(fixture_name, opts) do
    code_paths =
      :code.get_path()
      |> Enum.filter(&File.dir?/1)
      |> Enum.flat_map(&["-pa", List.to_string(&1)])

    num_warps = Keyword.get(opts, :num_warps, 4)

    expression = """
    Application.ensure_all_started(:beaver)
    result = Beaver.Shadow.Probe.evaluate(:#{fixture_name}, num_warps: #{num_warps})
    IO.puts("SHADOW_PROBE " <> JSON.encode!(result))
    """

    ["-e", expression | code_paths]
  end

  @doc """
  Evaluates one fixture in-process (used by the `mix shadow.probe` child).

  Updates `@marker_env` before each pipeline pass so a native crash can be
  attributed to the pass that was running.
  """
  @spec evaluate(atom(), keyword()) :: map()
  def evaluate(fixture_name, opts \\ []) when is_atom(fixture_name) do
    fixture = Corpus.fixture(fixture_name)
    num_warps = Keyword.get(opts, :num_warps, 4)
    context = MLIR.Context.create(all_dialects: false)

    try do
      Beaver.Triton.register(context)

      module =
        MLIR.Module.create!(File.read!(Corpus.fixture_path(fixture.name)), ctx: context)

      pipeline = lowering_pipeline(fixture.dialect, num_warps)

      lowered =
        Enum.reduce_while(pipeline, module, fn pass, acc ->
          mark(pass)

          try do
            {:cont, acc |> Beaver.Composer.append(pass) |> Beaver.Composer.run!()}
          rescue
            exception ->
              {:halt, {:pipeline_error, pass, Exception.message(exception)}}
          end
        end)

      case lowered do
        {:pipeline_error, _pass, message} ->
          %{status: "error", message: message}

        llvm ->
          mark("done")
          %{status: "ok", llvm_func: MLIR.to_string(llvm) =~ "llvm.func"}
      end
    rescue
      exception ->
        %{status: "error", message: Exception.message(exception)}
    after
      MLIR.Context.destroy(context)
    end
  end

  defp lowering_pipeline(dialect, num_warps) do
    convert =
      case dialect do
        :ttir -> "convert-triton-to-tritongpu{target=cuda:80, num-warps=#{num_warps}}"
        :ttgir -> nil
      end

    [
      convert,
      "tritongpu-coalesce",
      "tritongpu-F32DotTC",
      "triton-nvidia-gpu-plan-cta",
      "tritongpu-remove-layout-conversions",
      "tritongpu-optimize-thread-locality",
      "tritongpu-accelerate-matmul",
      "tritongpu-remove-layout-conversions",
      "tritongpu-optimize-dot-operands",
      "canonicalize",
      "tritongpu-combine-tensor-select-and-if",
      "tritongpu-allocate-warp-groups",
      "convert-scf-to-cf",
      "allocate-shared-memory-nv",
      "triton-tensor-memory-allocation",
      "triton-nvidia-check-matmul-two-cta",
      "triton-nvidia-gpu-proxy-fence-insertion",
      "triton-nvidia-gpu-tmem-barrier-insertion",
      "convert-triton-gpu-to-llvm",
      "convert-warp-specialize-to-llvm",
      "convert-nv-gpu-to-llvm",
      "convert-nvvm-to-llvm",
      "canonicalize"
    ]
    |> Enum.reject(&is_nil/1)
  end

  defp marker_path do
    Path.join(System.tmp_dir!(), "shadow_probe_#{System.unique_integer([:positive])}.marker")
  end

  defp mark(pass) do
    case System.get_env(@marker_env) do
      nil -> :ok
      path -> File.write!(path, pass <> "\n", [:append])
    end
  end

  defp last_pass(marker) do
    case File.read(marker) do
      {:ok, ""} -> nil
      {:ok, content} -> content |> String.split("\n", trim: true) |> List.last()
      {:error, _} -> nil
    end
  end

  defp parse_child_output(output) do
    output
    |> String.split("\n")
    |> Enum.find_value(fn line ->
      if String.starts_with?(line, @child_prefix) do
        payload =
          binary_part(line, byte_size(@child_prefix), byte_size(line) - byte_size(@child_prefix))

        JSON.decode!(payload)
      else
        nil
      end
    end)
  end
end
