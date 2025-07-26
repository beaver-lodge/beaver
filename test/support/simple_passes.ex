defmodule TestPass do
  @moduledoc false
  alias Beaver.MLIR
  use MLIR.Pass

  def run(%Beaver.MLIR.Operation{} = op, state) do
    MLIR.verify!(op)
    state
  end
end

defmodule TestFuncPass do
  @moduledoc false
  alias Beaver.MLIR
  use MLIR.Pass, on: "func.func"

  def run(%Beaver.MLIR.Operation{} = op, state) do
    MLIR.verify!(op)
    state
  end
end

defmodule ErrInit do
  @moduledoc false
  use Beaver.MLIR.Pass, on: "func.func"

  def initialize(_ctx, nil) do
    {:error, "new state used in cleanup"}
  end

  def destruct(state) do
    {:ok, _} = Agent.start_link(fn -> state end, name: __MODULE__)
    :ok
  end
end

defmodule IncorrectInitReturns do
  @moduledoc false
  use Beaver.MLIR.Pass, on: "func.func"

  def initialize(_ctx, nil) do
    :some
  end

  def destruct(state) do
    {:ok, _} = Agent.start_link(fn -> state end, name: __MODULE__)
    :ok
  end
end
