defmodule Beaver.Build.Plan do
  @moduledoc "Immutable stages in dependency order; callbacks receive named dependencies and execution records timings."
  defstruct stages: []
  def new, do: %__MODULE__{}

  def stage(%__MODULE__{stages: stages} = plan, name, dependencies, run) when is_function(run, 1) do
    known = MapSet.new(stages, & &1.name)
    if MapSet.member?(known, name), do: raise(ArgumentError, "duplicate build stage: #{inspect(name)}")
    missing = Enum.reject(dependencies, &MapSet.member?(known, &1))
    if missing != [], do: raise(ArgumentError, "unknown stage dependencies: #{inspect(missing)}")
    %{plan | stages: stages ++ [%{name: name, dependencies: dependencies, run: run}]}
  end

  @doc "Returns the dependency graph without executable callbacks."
  def describe(%__MODULE__{stages: stages}), do: Enum.map(stages, &Map.take(&1, [:name, :dependencies]))

  @doc "Runs a plan and publishes a selected stage's JSON report with total and per-stage milliseconds."
  def report!(plan, path, stage \\ :publish) do
    start = System.monotonic_time(:nanosecond)
    result = run!(plan)
    elapsed = (System.monotonic_time(:nanosecond) - start) / 1.0e6
    report = Map.merge(Map.fetch!(result.results, stage), %{"elapsed_ms" => elapsed, "stages_ms" => result.timings})
    Beaver.Build.ArtifactCache.write_atomic!(path, JSON.encode!(report))
    report
  end

  def run!(%__MODULE__{stages: stages}) do
    Enum.reduce(stages, %{results: %{}, timings: %{}}, fn stage, state ->
      start = System.monotonic_time(:nanosecond)
      value = stage.run.(Map.take(state.results, stage.dependencies))
      elapsed = (System.monotonic_time(:nanosecond) - start) / 1.0e6
      %{results: Map.put(state.results, stage.name, value), timings: Map.put(state.timings, stage.name, elapsed)}
    end)
  end
end
