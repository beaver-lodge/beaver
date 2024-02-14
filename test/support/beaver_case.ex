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
        diagnostic_server = unquote(options)[:diagnostic_server]
        ctx = MLIR.Context.create(diagnostic_server: diagnostic_server)
        require Logger

        on_exit(fn ->
          MLIR.Context.destroy(ctx)
        end)

        [ctx: ctx]
      end
    end
  end
end
