defmodule Beaver.Case do
  @moduledoc """
  Test case for beaver tests, with a MLIR context created and destroy automatically.
  """
  require Logger

  use ExUnit.CaseTemplate

  using do
    quote do
      alias Beaver.MLIR

      setup do
        {:ok, pid} = GenServer.start_link(Beaver.Diagnostic.Server, [])
        # pid = nil
        ctx = MLIR.Context.create(diagnostic_server: pid)
        require Logger

        on_exit(fn ->
          MLIR.CAPI.mlirContextDestroy(ctx)
        end)

        [ctx: ctx]
      end
    end
  end
end
