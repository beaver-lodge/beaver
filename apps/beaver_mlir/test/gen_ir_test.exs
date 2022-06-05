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
        Builtin.module do
          Func.func some_func() do
            region do
              block bb1() do
                v0 = Arith.constant({:value, ~a{0: i32}}) :: ~t<i32>
                cond0 = Arith.constant(true)

                CF.cond_br(cond0, :bb1, {:bb2, [v0]})
              end

              block bb1() do
                v1 = Arith.constant({:value, ~a{0: i32}}) :: ~t<i32>
                CF.br({:bb2, [v1]})
              end

              block bb2(arg :: ~t<i32>) do
                v2 = Arith.constant({:value, ~a{0: i32}}) :: ~t<i32>
                add = Arith.addi(arg, v2) :: ~t<i32>
                Func.return(add)
              end
            end
          end
        end
      end
    end

    Toy.gen_some_ir()
    |> MLIR.Operation.dump()
    |> MLIR.Operation.verify!()
  end
end
