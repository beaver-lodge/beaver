defmodule Mix.Tasks.Shadow.TuneReport do
  use Mix.Task

  @shortdoc "Emits a Shadow Wavefront tuning measurement report as Markdown"

  @moduledoc """
  Measures a corpus fixture across a config space and prints a paste-ready
  Markdown report (environment facts, records, winner, per-config crash
  probes).

      $ mix shadow.tune_report matmul

  Requires a Triton-enabled native build (`BEAVER_TRITON_PREBUILT_DIR` set).
  """

  @impl true
  def run([fixture_name]) do
    Mix.Task.run("app.start")

    fixture = fixture_name |> String.to_existing_atom() |> Beaver.Shadow.Corpus.fixture()
    configs = default_configs()

    fixture.name
    |> Beaver.Shadow.TuneReport.generate(configs)
    |> Beaver.Shadow.TuneReport.render()
    |> IO.puts()
  end

  def run(_) do
    IO.puts("usage: mix shadow.tune_report <fixture>")
  end

  defp default_configs do
    [
      %Beaver.Shadow.Tuning.Config{index: 0, num_warps: 2},
      %Beaver.Shadow.Tuning.Config{index: 1, num_warps: 4},
      %Beaver.Shadow.Tuning.Config{index: 2, num_warps: 8}
    ]
  end
end
