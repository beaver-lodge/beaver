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
        ctx = MLIR.Context.create(unquote(options))

        {server, handler_id} =
          if unquote(options)[:diagnostic] == :server do
            {:ok, pid} = GenServer.start(Beaver.Diagnostic.Server, [])
            {pid, Beaver.Diagnostic.attach(ctx, pid)}
          else
            {nil, Beaver.Diagnostic.attach(ctx)}
          end

        on_exit(fn ->
          if server do
            :ok = GenServer.stop(server)
          end

          Beaver.Diagnostic.detach(ctx, handler_id)
          MLIR.Context.destroy(ctx)
        end)

        [ctx: ctx, diagnostic_server: server]
      end
    end
  end
end
