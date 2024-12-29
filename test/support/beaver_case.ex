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

        on_exit(fn ->
          MLIR.Context.destroy(ctx)
        end)

        %{ctx: ctx}
      end
    end
  end
end
