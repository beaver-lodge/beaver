defmodule GenIRTest do
  use ExUnit.Case
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  test "example from upstream with br" do
    content = File.read!("test/br_example.mlir")

    ~m"""
    #{content}
    """
  end

  test "generate mlir with function calls" do
    defmodule Toy do
      alias Beaver.MLIR
      alias Beaver.MLIR.Dialect.{Builtin, Func, Arith, CF}
      import Builtin, only: :macros
      import Func, only: :macros
      import MLIR, only: :macros
      import MLIR.Sigils

      def gen_some_ir() do
        Beaver.MLIR.Dialect.Builtin.module do
          Beaver.MLIR.Dialect.Func.func some_func() :: ~t<i32> do
            v0 = Arith.constant(0) :: ~t<i32>
            cond0 = Arith.constant(true)
            CF.cond_br(cond0, :bb1, {:bb2, [v0]})

            MLIR.block bb1() do
              v1 = Arith.constant(1) :: ~t<i32>
              CF.br({:bb2, v1})
            end

            MLIR.block bb2(arg :: ~t<i32>) do
              v2 = Arith.constant(1) :: ~t<i32>
              add = Arith.addi(arg, v2) :: ~t<i32>
              Func.return(add) :: ~t<i32>
            end
          end
        end
      end
    end

    Toy.gen_some_ir()
  end
end
