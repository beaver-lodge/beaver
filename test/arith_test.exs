defmodule ArithTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Type
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func
  @moduletag :smoke

  test "int predicates", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        for p <- [:eq, :ne, :slt, :sle, :sgt, :sge, :ult, :ule, :ugt, :uge] do
          Func.func _(
                      function_type: Type.function([Type.i32(), Type.i32()], []),
                      sym_name:
                        Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                    ) do
            region do
              block _(a >>> Type.i32(), b >>> Type.i32()) do
                Arith.cmpi(a, b, predicate: Arith.cmp_i_predicate(p)) >>> Type.i1()
                Func.return() >>> []
              end
            end
          end
          |> tap(&assert(MLIR.to_string(&1, generic: false) =~ "#{p}"))
        end
      end
      |> MLIR.verify!()
    end
  end

  for p <- [:!=, :==, :>, :>=, :<, :<=], signed <- [true, false] do
    test "int operator #{p}, signed: #{signed}", %{ctx: ctx} do
      mlir ctx: ctx do
        module do
          Func.func _(
                      function_type: Type.function([Type.i32(), Type.i32()], []),
                      sym_name:
                        Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                    ) do
            region do
              block _(a >>> Type.i32(), b >>> Type.i32()) do
                Arith.cmpi(a, b,
                  predicate: Arith.operator_to_predicate(unquote(p), unquote(signed))
                ) >>>
                  Type.i1()

                Func.return() >>> []
              end
            end
          end
        end
      end
      |> MLIR.verify!()
    end
  end

  test "float predicates", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        for p <- [
              false,
              :oeq,
              :ogt,
              :oge,
              :olt,
              :ole,
              :one,
              :ord,
              :ueq,
              :ugt,
              :uge,
              :ult,
              :ule,
              :une,
              :uno,
              true
            ] do
          f =
            Func.func _(
                        function_type: Type.function([Type.f32(), Type.f32()], []),
                        sym_name: MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                      ) do
              region do
                block _(a >>> Type.f32(), b >>> Type.f32()) do
                  Arith.cmpf(a, b, predicate: Arith.cmp_f_predicate(p)) >>> Type.i1()
                  Func.return() >>> []
                end
              end
            end

          assert f |> MLIR.to_string(generic: false) =~ "#{p},"
          refute f |> MLIR.to_string(generic: true) =~ "#{p},"
        end
      end
    end
    |> MLIR.verify!()
  end
end
