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

        ds_pid =
          case unquote(options)[:diagnostic] do
            :server ->
              {:ok, pid} = GenServer.start(Beaver.Diagnostic.Server, [])
              id = Beaver.Diagnostic.attach(ctx, pid)

              on_exit(fn ->
                # Beaver.Diagnostic.detach(ctx, id)
                :ok = GenServer.stop(pid)
              end)

              pid

            _ ->
              id = Beaver.Diagnostic.attach(ctx, :stderr)
              # on_exit(fn -> Beaver.Diagnostic.detach(ctx, id) end)
              nil
          end

        on_exit(fn ->
          MLIR.Context.destroy(ctx)
        end)

        [ctx: ctx, diagnostic_server: ds_pid]
      end
    end
  end
end
