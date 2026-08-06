defmodule Beaver.MLIR.Transform.Solver do
  @moduledoc """
  Optional adapter for inspecting or validating `transform.smt.constrain_params`.

  Beaver deliberately does not choose an SMT implementation. An adapter gets
  the exported constraint IR and the already resolved Tune selections. It may
  return deterministic metadata for recording alongside the resolved schedule.
  The adapter does not need to mutate MLIR objects.
  """

  alias Beaver.MLIR.Transform.Schedule.Constraint

  @callback solve([Constraint.t()], selections :: map(), state :: term()) ::
              {:ok, metadata :: term(), new_state :: term()}
              | {:error, reason :: term(), new_state :: term()}
end
