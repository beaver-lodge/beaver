defmodule Toy do
  alias Beaver.MLIR
  alias Beaver.MLIR.Builtin
  alias Beaver.MLIR.Dialect.{Func, Arith, CF}

  def gen_some_ir() do
    Builtin.module do
      Func.func some_func()  do
        v0 = Arith.constant(0) :: ~t<i32>
        cond0 = Arith.constant(true)
        CF.cond_br(cond0, :bb1, {:bb2, [v0]})

        MLIR.block bb1() do
          v1 = Arith.constant(1) :: ~t<i32>
          CF.br({:bb2, v1})
        end

        MLIR.block bb2(arg :: ~t<i32>) do
          v2 = Arith.constant(1) :: ~t<i32>
          Arith.addi(arg, v2) :: ~t<i32>
        end
      end
    end
    |> MLIR.Operation.verify!()
  end
end
