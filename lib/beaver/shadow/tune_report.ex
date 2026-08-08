defmodule Beaver.Shadow.TuneReport do
  @moduledoc """
  Renders a Shadow Wavefront tuning measurement as paste-ready Markdown.

  Combines the CPU-only tuning run (`Beaver.Shadow.Tuning.run/3`), the
  per-config crash probes (`Beaver.Shadow.Probe.probe_many/2`), and the
  environment facts (LLVM revision, Triton prebuilt dir, target) into a
  report that can be pasted into upstream Triton issues or RFC discussions.
  """

  alias Beaver.Shadow.Probe
  alias Beaver.Shadow.Tuning

  defstruct [
    :fixture,
    :generated_at,
    :llvm_revision,
    :prebuilt_dir,
    :target,
    :config_space_digest,
    :records,
    :winner,
    :probes
  ]

  @type t() :: %__MODULE__{
          fixture: atom(),
          generated_at: DateTime.t(),
          llvm_revision: String.t() | nil,
          prebuilt_dir: String.t() | nil,
          target: String.t(),
          config_space_digest: String.t(),
          records: [Tuning.Record.t()],
          winner: Tuning.Record.t() | nil,
          probes: [%{config: Tuning.Config.t(), result: Probe.Result.t()}]
        }

  @doc "Measures a config space and its crash probes for one corpus fixture."
  @spec generate(atom(), [Tuning.Config.t()], keyword()) :: t()
  def generate(fixture_name, configs, opts \\ []) when is_atom(fixture_name) do
    target = Keyword.get(opts, :target, "cuda:80")
    run = Tuning.run(fixture_name, configs, target: target)

    %__MODULE__{
      fixture: fixture_name,
      generated_at: DateTime.utc_now(),
      llvm_revision: llvm_revision(),
      prebuilt_dir: System.get_env("BEAVER_TRITON_PREBUILT_DIR"),
      target: target,
      config_space_digest: Tuning.configs_digest(configs),
      records: run.records,
      winner: run.winner,
      probes: Probe.probe_many(fixture_name, configs)
    }
  end

  @doc "Renders the measurement as Markdown."
  @spec render(t()) :: String.t()
  def render(%__MODULE__{} = report) do
    """
    ## Shadow Wavefront: Triton tuning measurement

    - fixture: `#{report.fixture}`
    - generated at: #{DateTime.to_iso8601(report.generated_at)}
    - llvm revision: #{report.llvm_revision || "unknown"}
    - triton prebuilt dir: `#{report.prebuilt_dir || "unset"}`
    - target: `#{report.target}`
    - config space digest: `#{report.config_space_digest}`

    ### Config space

    | index | num_warps | num_stages | num_ctas |
    |---|---|---|---|
    #{Enum.map_join(report.records, "\n", fn record -> config_row(record.config) end)}

    ### Records

    | index | status | baseline convert_layouts | optimized | reduction | lowers to LLVM | total_ns |
    |---|---|---|---|---|---|---|
    #{Enum.map_join(report.records, "\n", &record_row/1)}

    ### Winner

    #{winner_text(report.winner)}

    ### Per-config probes

    | index | num_warps | probe |
    |---|---|---|
    #{Enum.map_join(report.probes, "\n", &probe_row/1)}
    """
    |> String.trim_trailing()
    |> Kernel.<>("\n")
  end

  defp config_row(config) do
    "| #{config.index} | #{config.num_warps} | #{config.num_stages || "-"} | #{config.num_ctas || "-"} |"
  end

  defp record_row(record) do
    proxy = record.structural_proxy

    "| #{record.config.index} | #{record.status} | #{proxy_field(proxy, :baseline)} | " <>
      "#{proxy_field(proxy, :optimized)} | #{reduction(proxy)} | " <>
      "#{proxy_field(proxy, :lowered_to_llvm)} | " <>
      "#{(record.timings && record.timings.total_ns) || "-"} |"
  end

  defp proxy_field(nil, _field), do: "-"
  defp proxy_field(proxy, field), do: Map.get(proxy, field) || "-"

  defp reduction(%{baseline: baseline, optimized: optimized}) when baseline > 0,
    do: "-#{baseline - optimized}"

  defp reduction(_proxy), do: "-"

  defp winner_text(nil), do: "no winner (no config lowered to LLVM)"

  defp winner_text(winner) do
    proxy = winner.structural_proxy

    "config #{winner.config.index} (num_warps #{winner.config.num_warps}) with " <>
      "#{proxy.optimized} convert_layouts after optimization."
  end

  defp probe_row(%{config: config, result: result}) do
    text =
      case result.status do
        :ok -> "ok"
        :crash -> "crash (exit #{result.exit_code}) at `#{result.detail.last_pass}`"
        :error -> "error: `#{result.detail}`"
      end

    "| #{config.index} | #{config.num_warps} | #{text} |"
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
