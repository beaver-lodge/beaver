defmodule Mix.Tasks.Shadow.Probe do
  use Mix.Task

  @shortdoc "Runs a Triton corpus fixture through the lowering pipeline"

  @moduledoc """
  Probes one corpus fixture through the Triton lowering pipeline and prints a
  machine-readable outcome prefixed with `SHADOW_PROBE `.

  `Beaver.Shadow.Probe.run/1` drives the same evaluation in a child BEAM so
  a native crash in the Triton prebuilt cannot take down the caller; this
  task is the interactive form:

      $ mix shadow.probe remat
  """

  @impl true
  def run([fixture_name]) do
    Mix.Task.run("app.start")

    fixture_name
    |> String.to_existing_atom()
    |> Beaver.Shadow.Probe.evaluate()
    |> then(&IO.puts("SHADOW_PROBE " <> JSON.encode!(&1)))
  end
end
