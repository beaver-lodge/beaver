defmodule ArithTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Type
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func
  @moduletag :smoke

  test "cmp_i_predicate", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        for p <- [:eq, :ne, :slt, :sle, :sgt, :sge, :ult, :ule, :ugt, :uge] do
          f =
            Func.func _(
                        function_type: Type.function([Type.i32(), Type.i32()], []),
                        sym_name: "\"#{p}\""
                      ) do
              region do
                block _(a >>> Type.i32(), b >>> Type.i32()) do
                  Arith.cmpi(a, b, predicate: Arith.cmp_i_predicate(p)) >>> Type.i1()
                  Func.return() >>> []
                end
              end
            end

          assert f |> MLIR.to_string(generic: false) =~ "#{p}"
        end
      end
      |> MLIR.Operation.verify!()
    end
  end

  test "f_type_to_magic_num", test_context do
    mlir ctx: test_context[:ctx] do
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
                        sym_name: "\"#{p}\""
                      ) do
              region do
                block _(a >>> Type.f32(), b >>> Type.f32()) do
                  Arith.cmpf(a, b, predicate: Arith.cmp_f_predicate(p)) >>> Type.i1()
                  Func.return() >>> []
                end
              end
            end

          assert f |> MLIR.to_string(generic: false) =~ "#{p}"
        end
      end
    end
    |> MLIR.Operation.verify!()
  end
end
