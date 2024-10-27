defmodule Beaver.Case do
  @moduledoc """
  Test case for beaver tests, with a MLIR context created and destroy automatically.
  """

  use ExUnit.CaseTemplate

  using options do
    quote do
      alias Beaver.MLIR

      setup do
        Beaver.MLIR.Pass.ensure_all_registered!()
        ctx = MLIR.Context.create(unquote(options))

        {server, handler_id} =
          if unquote(options)[:diagnostic] == :server do
            {:ok, pid} =
              GenServer.start(
                Beaver.DiagnosticsCapturer,
                &"#{&2}[Beaver] [Diagnostic] [#{to_string(MLIR.location(&1))}] #{to_string(&1)}\n"
              )

            {pid, Beaver.DiagnosticsCapturer.attach(ctx, pid)}
          else
            {nil, nil}
          end

        on_exit(fn ->
          if server do
            :ok = GenServer.stop(server)
            Beaver.MLIR.Diagnostic.detach(ctx, handler_id)
          end

          MLIR.Context.destroy(ctx)
        end)

        %{ctx: ctx, diagnostic_server: server}
      end
    end
  end
end
