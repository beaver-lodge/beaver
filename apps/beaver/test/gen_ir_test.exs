defmodule GenIRTest do
  use ExUnit.Case
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  test "generate mlir with function calls" do
    require Beaver
    alias Beaver.MLIR
    alias Beaver.MLIR.Dialect.{Builtin, Func, Arith, CF}
    import Builtin, only: :macros
    import Func, only: :macros
    import MLIR, only: :macros
    import MLIR.Sigils

    Beaver.mlir do
      Builtin.module do
        Func.func some_func(function_type: ~a"() -> i32") do
          region do
            block bb_entry() do
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
    |> MLIR.Operation.dump()
    |> MLIR.Operation.verify!()
  end

  test "generate mlir with function calls ast" do
    require Beaver
    alias Beaver.MLIR
    alias Beaver.MLIR.Dialect.{Builtin, Func, Arith, CF}
    import Builtin, only: :macros
    import Func, only: :macros
    import MLIR, only: :macros
    import MLIR.Sigils

    ast =
      quote do
        Beaver.mlir do
          Builtin.module do
            Func.func some_func(function_type: ~a"() -> i32") do
              region do
                block bb_entry() do
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

    env = __ENV__
    Macro.prewalk(ast, &Macro.expand(&1, env)) |> Macro.to_string() |> IO.puts()
  end
end
