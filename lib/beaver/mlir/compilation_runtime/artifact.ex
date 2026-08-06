defmodule Beaver.MLIR.CompilationRuntime.Artifact do
  @moduledoc "A reusable, version-checked MLIR bytecode compilation artifact."

  @enforce_keys [:key, :bytecode, :metadata, :cache, :timings]
  defstruct [:key, :bytecode, :metadata, :cache, :timings]

  @type t :: %__MODULE__{
          key: String.t(),
          bytecode: binary(),
          metadata: map(),
          cache: :hit | :miss,
          timings: %{optional(atom()) => integer()}
        }
end
