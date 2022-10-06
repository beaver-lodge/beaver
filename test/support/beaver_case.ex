defmodule Beaver.Case do
  @moduledoc """
  Test case for beaver tests, with a MLIR context created and destroy automatically.
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      alias Beaver.MLIR

      setup do
        ctx = MLIR.Context.create()

        on_exit(fn ->
          MLIR.CAPI.mlirContextDestroy(ctx)
        end)

        [ctx: ctx]
      end
    end
  end
end
