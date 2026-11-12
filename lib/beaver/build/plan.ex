defmodule Beaver.Build.Plan do
  @moduledoc """
  An immutable build plan with explicit stage dependencies and measured execution.

  Stages are declared in dependency order. Each callback receives only its named
  dependencies; IO and caching remain in the stage, so plans are inspectable before
  execution. Independent compilation units can use LLVMIR's concurrent executor.
  """
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
