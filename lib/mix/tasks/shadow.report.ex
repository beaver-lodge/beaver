defmodule Mix.Tasks.Shadow.Report do
  use Mix.Task

  @shortdoc "Emits a Shadow Wavefront Triton measurement report as Markdown"

  @moduledoc """
  Measures the Triton corpus and prints a Markdown report that can be pasted
  into upstream issues or pull requests.

      $ mix shadow.report

  Requires a Triton-enabled native build (`BEAVER_TRITON_PREBUILT_DIR` set).
  """

  @impl true
  def run(_args) do
    Mix.Task.run("app.start")
    Beaver.Shadow.Report.generate() |> Beaver.Shadow.Report.render() |> IO.puts()
  end
end
