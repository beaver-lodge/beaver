defmodule Beaver.MLIR.Transform.Resolver do
  @moduledoc """
  Behaviour for resolving Transform Tune choices.

  Resolver state is explicit so one resolver may be reused safely by concurrent
  tuning candidates. Returning `:unresolved` lets an already selected value be
  used; otherwise resolution fails for an active choice.
  """

  alias Beaver.MLIR.Transform.Schedule.Choice

  @callback resolve(Choice.t(), state :: term()) ::
              {:ok, term(), new_state :: term()}
              | {:error, term(), new_state :: term()}
              | {:unresolved, new_state :: term()}
end
