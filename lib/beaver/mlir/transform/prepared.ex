defmodule Beaver.MLIR.Transform.Prepared do
  @moduledoc """
  A validated schedule and resolved entry sequence owned by the preparing process. Reuse with
  `Transform.apply_named_sequence/3`; close before destroying its context. Each application has independent MLIR
  transform execution state.
  """
  @enforce_keys [:ref, :context, :sequence, :owner]
  defstruct [:ref, :context, :sequence, :owner]
  @type t :: %__MODULE__{}
end
