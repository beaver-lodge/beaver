defmodule Beaver.Slang do
  @moduledoc """
  Defining a MLIR dialect with macros in Elixir. Internally expressions are compiled to [IRDL](https://mlir.llvm.org/docs/Dialects/IRDL/)
  """
  defmacro __using__(_) do
    quote do
      import Beaver.Slang
    end
  end

  defmacro deftype(call, block) do
    {name, args} =
      call
      |> Macro.decompose_call()
      |> dbg()

    quote do
      def unquote(name)(_) do
      end
    end
  end

  defmacro defop(_) do
  end

  defmacro defop(_, _) do
  end

  def any_of(_) do
  end

  def any() do
  end
end
