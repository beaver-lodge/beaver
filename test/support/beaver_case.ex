defmodule Beaver.Case do
  @moduledoc "Test case for beaver tests, with a MLIR context created and destroy automatically."

  use ExUnit.CaseTemplate

  using options do
    quote do
      use Beaver
      alias Beaver.MLIR
      require Beaver.MLIR.Dialect.Func
      import Beaver.Case, only: [function_module: 2]

      setup do
        ctx = MLIR.Context.create(unquote(options))
        on_exit(fn -> MLIR.Context.destroy(ctx) end)
        %{ctx: ctx}
      end
    end
  end

  @doc "Wraps the caller's block DSL in a module, func.func and region, retaining its enclosing mlir environment."
  defmacro function_module(signature, do: body) do
    quote do
      use Beaver
      alias Beaver.MLIR
      require Beaver.MLIR.Dialect.Func

      module do
        Beaver.MLIR.Dialect.Func.func unquote(signature) do
          region do
            unquote(body)
          end
        end
      end
    end
  end
end
