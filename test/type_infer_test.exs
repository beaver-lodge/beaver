defmodule TypeInferTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func
  @moduletag :smoke

  test "type infer", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        width = 8
        vector_t = ~t{vector<#{width}xi32>}

        value =
          1..width
          |> Enum.map(&Attribute.integer(Type.i32(), &1))
          |> Attribute.dense_elements(Type.vector!([width], Type.i32(ctx: ctx)))

        Func.func some_func(function_type: Type.function([], [vector_t])) do
          region do
            block do
              v = Arith.constant(value: value) >>> vector_t
              v0 = Arith.addi(v, v) >>> :infer
              assert v0 |> MLIR.Value.type() |> MLIR.equal?(vector_t)
              Func.return(v0) >>> []
            end
          end
        end
      end
    end
    |> MLIR.verify!()
    |> MLIR.Transform.canonicalize()
    |> Beaver.Composer.run!()
  end

  test "set to infer but given types", %{ctx: ctx} do
    assert_raise ArgumentError, "already set to infer the result types", fn ->
      mlir ctx: ctx do
        module do
          Func.func some_func(function_type: Type.function([], [])) do
            region do
              block do
                v = Arith.constant(value: Attribute.integer(Type.i32(), 1)) >>> ~t{i32}
                Arith.addi(v, v) >>> [:infer, ~t{i32}]
              end
            end
          end
        end
      end
    end
  end
end
