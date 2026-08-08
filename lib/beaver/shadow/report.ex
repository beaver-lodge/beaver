defmodule Beaver.Shadow.Report do
  @moduledoc """
  Gathers Shadow Wavefront measurement facts for the Triton corpus and renders
  them as Markdown ready to paste into upstream issues or pull requests.

  TTIR fixtures are measured through `Beaver.Shadow.OptimizationTrial`
  (baseline/optimized `ttg.convert_layout` counts and whether they lower to
  LLVM). TTGIR fixtures are audited in place and probed through
  `Beaver.Shadow.Probe`, so a pinned-prebuilt crash shows up as a failure
  receipt with the offending pass.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.Triton.LayoutAudit
  alias Beaver.Shadow.{Corpus, OptimizationTrial, Probe}

  defstruct [:generated_at, :llvm_revision, :prebuilt_dir, :fixtures]

  @type fixture_measure() :: %{
          name: atom(),
          dialect: :ttir | :ttgir,
          baseline: non_neg_integer(),
          optimized: non_neg_integer(),
          lowered_to_llvm: boolean() | nil,
          probe: Probe.Result.t() | nil
        }

  @type t() :: %__MODULE__{
          generated_at: DateTime.t(),
          llvm_revision: String.t() | nil,
          prebuilt_dir: String.t() | nil,
          fixtures: [fixture_measure()]
        }

  @doc """
  Measures every corpus fixture and returns the collected facts.

  Requires a Triton-enabled native build (`BEAVER_TRITON_PREBUILT_DIR`).
  """
  @spec generate() :: t()
  def generate do
    %__MODULE__{
      generated_at: DateTime.utc_now(),
      llvm_revision: llvm_revision(),
      prebuilt_dir: System.get_env("BEAVER_TRITON_PREBUILT_DIR"),
      fixtures: Enum.map(Corpus.fixtures(), &measure/1)
    }
  end

  @doc "Renders the collected facts as a Markdown report."
  @spec render(t()) :: String.t()
  def render(%__MODULE__{} = report) do
    """
    ## Shadow Wavefront: Triton layout measurement

    - generated at: #{DateTime.to_iso8601(report.generated_at)}
    - llvm revision: #{report.llvm_revision || "unknown"}
    - triton prebuilt dir: `#{report.prebuilt_dir || "unset"}`

    | fixture | dialect | baseline convert_layouts | optimized | reduction | lowers to LLVM | probe |
    |---|---|---|---|---|---|---|
    #{Enum.map_join(report.fixtures, "\n", &row/1)}
    """
    |> String.trim_trailing()
    |> Kernel.<>("\n")
  end

  defp row(fixture) do
    reduction =
      case {fixture.baseline, fixture.optimized} do
        {baseline, optimized} when baseline > 0 -> "-#{baseline - optimized}"
        _ -> "-"
      end

    lowered =
      case fixture.lowered_to_llvm do
        true -> "yes"
        false -> "no"
        nil -> "-"
      end

    probe =
      probe_text(fixture.probe)

    "| #{fixture.name} | #{fixture.dialect} | #{fixture.baseline} | #{fixture.optimized} | " <>
      "#{reduction} | #{lowered} | #{probe} |"
  end

  defp probe_text(%Probe.Result{status: :crash} = probe) do
    "crash (exit #{probe.exit_code}) at `#{probe.detail.last_pass}`"
  end

  defp probe_text(%Probe.Result{status: :ok}), do: "ok"

  defp probe_text(%Probe.Result{status: :error, detail: message}), do: "error: `#{message}`"

  defp probe_text(nil), do: "-"

  defp measure(%{dialect: :ttir} = fixture) do
    context = MLIR.Context.create(all_dialects: false)

    result =
      try do
        Beaver.Triton.register(context)

        module = MLIR.Module.create!(File.read!(Corpus.fixture_path(fixture.name)), ctx: context)
        trial = OptimizationTrial.run(module)

        %{
          name: fixture.name,
          dialect: :ttir,
          baseline: trial.baseline,
          optimized: trial.optimized,
          lowered_to_llvm: trial.lowered_to_llvm,
          probe: nil
        }
      after
        MLIR.Context.destroy(context)
      end

    result
  end

  defp measure(%{dialect: :ttgir} = fixture) do
    context = MLIR.Context.create(all_dialects: false)

    counts =
      try do
        Beaver.Triton.register(context)

        module = MLIR.Module.create!(File.read!(Corpus.fixture_path(fixture.name)), ctx: context)
        baseline = LayoutAudit.audit(module).operation_count

        optimized =
          module
          |> Beaver.Composer.append("tritongpu-remove-layout-conversions")
          |> Beaver.Composer.run!()
          |> LayoutAudit.audit()
          |> Map.fetch!(:operation_count)

        {baseline, optimized}
      after
        MLIR.Context.destroy(context)
      end

    {baseline, optimized} = counts

    %{
      name: fixture.name,
      dialect: :ttgir,
      baseline: baseline,
      optimized: optimized,
      lowered_to_llvm: false,
      probe: Probe.run(fixture.name)
    }
  end

  defp llvm_revision do
    case System.get_env("LLVM_CONFIG_PATH") do
      nil ->
        nil

      llvm_config ->
        case System.cmd(llvm_config, ["--version"], stderr_to_stdout: true) do
          {output, 0} -> output |> String.split("\n") |> hd()
          _ -> nil
        end
    end
  end
end
