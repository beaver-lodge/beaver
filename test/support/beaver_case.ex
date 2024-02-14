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
          case unquote(options)[:diagnostic] do
            :server ->
              {:ok, pid} = GenServer.start(Beaver.Diagnostic.Server, [])
              id = Beaver.Diagnostic.attach(ctx, pid)
              {pid, id}

            _ ->
              id = Beaver.Diagnostic.attach(ctx, :stderr)
              {nil, id}
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
