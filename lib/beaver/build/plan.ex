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

  def run!(%__MODULE__{stages: stages}) do
    Enum.reduce(stages, %{results: %{}, timings: %{}}, fn stage, state ->
      start = System.monotonic_time(:nanosecond)
      value = stage.run.(Map.take(state.results, stage.dependencies))
      elapsed = (System.monotonic_time(:nanosecond) - start) / 1.0e6
      %{results: Map.put(state.results, stage.name, value), timings: Map.put(state.timings, stage.name, elapsed)}
    end)
  end
end
