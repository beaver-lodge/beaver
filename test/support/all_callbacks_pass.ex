defmodule AllCallbacks do
  @moduledoc false
  alias Beaver.MLIR
  use Beaver.MLIR.Pass, on: "func.func"

  @counter %{
    destruct: 0,
    initialize: 0,
    clone: 0,
    run: 0
  }
  def initialize(_ctx, nil) do
    {:ok, pid} =
      Agent.start_link(
        fn ->
          %{@counter | initialize: 1}
        end,
        name: __MODULE__
      )

    {:ok, {:init, pid}}
  end

  def initialize(_ctx, {:init, pid}) do
    Agent.update(
      pid,
      fn state ->
        update_in(state.initialize, &(&1 + 1))
      end
    )

    {:ok, {:re_init, pid}}
  end

  def clone({:init, pid}) do
    Agent.update(pid, fn state -> update_in(state.clone, &(&1 + 1)) end)
    {:clone, pid}
  end

  def run(op, {:clone, pid} = state) do
    "func.func" = MLIR.Operation.name(op)
    :ok = Agent.update(pid, fn state -> update_in(state.run, &(&1 + 1)) end)
    state
  end

  def destruct({:clone, pid}) do
    Agent.update(pid, fn state -> update_in(state.destruct, &(&1 + 1)) end)
  end

  def destruct({:init, pid}) do
    Agent.update(pid, fn state -> update_in(state.destruct, &(&1 + 1)) end)
  end

  def destruct({:re_init, pid}) do
    Agent.update(pid, fn state -> update_in(state.destruct, &(&1 + 1)) end)
  end
end
