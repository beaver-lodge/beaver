defmodule Beaver.Charm do
  use Beaver

  defmodule CompilerCtx do
    @moduledoc false
    defstruct ctx: nil
  end

  defp gen_mlir(ssa, acc) do
    ssa |> dbg
    {ssa, acc}
  end

  def compile(ssa) do
    ctx = MLIR.Context.create()
    Intermediator.SSA.peek(ssa)
    Intermediator.SSA.prewalk(ssa, %CompilerCtx{ctx: ctx}, &gen_mlir/2)
  end

  def run(_mod) do
  end
end
