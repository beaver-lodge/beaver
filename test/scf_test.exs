defmodule SCFTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.{Arith, Func, SCF}

  require Func

  @moduletag :smoke

  test "builds if regions and yields branch results", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
          Func.func choose(function_type: Type.function([Type.i1()], [Type.i32()])) do
            region do
              block _entry(condition >>> Type.i1()) do
                value =
                  SCF.if condition do
                    region do
                      block do
                        value =
                          Arith.constant(value: Attribute.integer(Type.i32(), 1)) >>> Type.i32()

                        SCF.yield(value) >>> []
                      end
                    end

                    region do
                      block do
                        value =
                          Arith.constant(value: Attribute.integer(Type.i32(), 2)) >>> Type.i32()

                        SCF.yield(value) >>> []
                      end
                    end
                  end >>> Type.i32()

                Func.return(value) >>> []
              end
            end
          end
        end
      end

    MLIR.verify!(module)
    assert to_string(module) =~ "scf.if"
    assert to_string(module) =~ "else"
  end

  test "builds for with loop-carried values", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
          Func.func sum(function_type: Type.function([], [Type.index()])) do
            region do
              block do
                lower = Arith.constant(value: Attribute.index(0)) >>> Type.index()
                upper = Arith.constant(value: Attribute.index(4)) >>> Type.index()
                step = Arith.constant(value: Attribute.index(1)) >>> Type.index()
                initial = Arith.constant(value: Attribute.index(0)) >>> Type.index()

                sum =
                  SCF.for [lower, upper, step, initial] do
                    region do
                      block _body(iv >>> Type.index(), acc >>> Type.index()) do
                        sum = Arith.addi(acc, iv) >>> Type.index()
                        SCF.yield(sum) >>> []
                      end
                    end
                  end >>> Type.index()

                Func.return(sum) >>> []
              end
            end
          end
        end
      end

    MLIR.verify!(module)
    assert to_string(module) =~ "scf.for"
    assert to_string(module) =~ "iter_args"
  end
end
