defmodule SCFTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.{Arith, Func, SCF}

  require Func
  require SCF

  @moduletag :smoke

  test "builds if regions and yields branch results", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
          Func.func choose(function_type: Type.function([Type.i1()], [Type.i32()])) do
            region do
              block _entry(condition >>> Type.i1()) do
                value =
                  SCF.if_ condition, result_types: [Type.i32()] do
                    Arith.constant(value: Attribute.integer(Type.i32(), 1)) >>> Type.i32()
                  else
                    Arith.constant(value: Attribute.integer(Type.i32(), 2)) >>> Type.i32()
                  end

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
                  SCF.for_ lower, upper, step, iter_args: [initial] do
                    fn iv, [acc] -> Arith.addi(acc, iv) >>> Type.index() end
                  end

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

  test "rejects a result-bearing if without else" do
    assert_raise ArgumentError, ~r/requires an else branch/, fn ->
      Code.compile_string("""
      defmodule InvalidResultBearingSCFIf do
        require Beaver.MLIR.Dialect.SCF

        def build do
          Beaver.MLIR.Dialect.SCF.if_ true, result_types: [:result] do
            :value
          end
        end
      end
      """)
    end
  end
end
