defmodule Beaver.Case do
  @moduledoc """
  Test case for beaver tests, with a MLIR context created and destroy automatically.
  """
  require Logger

  use ExUnit.CaseTemplate

  using options do
    quote do
      alias Beaver.MLIR

      setup do
        ctx = MLIR.Context.create(diagnostic: unquote(options)[:diagnostic])
        require Logger

        on_exit(fn ->
          MLIR.Context.destroy(ctx)
        end)

        [ctx: ctx]
      end
    end
  end
end
