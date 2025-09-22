defmodule ODSDumpTest do
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Type
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func
  use Beaver.Case, async: true

  @moduletag :smoke
  test "lookup" do
    assert {:ok,
            %{
              "attributes" => _,
              "operands" => _,
              "results" => _
            }} = MLIR.ODS.Dump.lookup("affine.for")

    assert {:error, "failed to find ODS dump of \"???\""} = MLIR.ODS.Dump.lookup("???")
  end

  test "tagged operands", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        Func.func _(
                    function_type: Type.function([Type.i32(), Type.i32()], []),
                    sym_name:
                      Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                  ) do
          region do
            block _(a >>> Type.i32(), b >>> Type.i32()) do
              Arith.addi(lhs: a, rhs: b) >>> Type.i32()
              Func.return() >>> []
            end
          end
        end
      end
      |> MLIR.verify!()
    end
  end

  test "tagged and untagged operands mixed", %{ctx: ctx} do
    mlir ctx: ctx do
      assert_raise ArgumentError, ~r"Cannot mix tagged and untagged operands", fn ->
        mlir ctx: ctx do
          module do
            Func.func _(
                        function_type: Type.function([Type.i32(), Type.i32()], []),
                        sym_name:
                          Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                      ) do
              region do
                block _(a >>> Type.i32(), b >>> Type.i32()) do
                  Arith.addi(a, lhs: a, rhs: b) >>> Type.i32()
                  Func.return() >>> []
                end
              end
            end
          end
        end
      end
    end
  end

  test "tagged operands unconsumed", %{ctx: ctx} do
    import ExUnit.CaptureLog

    mlir ctx: ctx do
      logs =
        capture_log(fn ->
          mlir ctx: ctx do
            module do
              Func.func _(
                          function_type: Type.function([Type.i32(), Type.i32()], []),
                          sym_name:
                            Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                        ) do
                region do
                  block _(_a >>> Type.i32(), b >>> Type.i32()) do
                    Arith.addi(rhs: b) >>> Type.i32()
                    Func.return() >>> []
                  end
                end
              end
            end
          end
        end)

      assert logs =~ ~r"Single operand 'lhs' not set when creating operation arith\.addi"
    end
  end

  test "segment_sizes without ods dump", %{ctx: ctx} do
    mlir ctx: ctx do
      MLIR.Context.allow_unregistered_dialects(ctx)

      m =
        module do
          a = Arith.constant(value: Attribute.integer(Type.i32(), 1)) >>> :infer
          b = Arith.constant(value: Attribute.integer(Type.i32(), 2)) >>> :infer
          UndefinedDialect.foo(lhs: a, rhs: b, operand_segment_sizes: :infer) >>> Type.i32()

          UndefinedDialect.foo(rhs: [a], lhs: [a, b], operand_segment_sizes: :infer) >>>
            Type.i32()
        end

      ops = MLIR.Module.body(m) |> Beaver.Walker.operations()
      assert 1 = ops[2][:operand_segment_sizes][0]
      assert 1 = ops[2][:operand_segment_sizes][1]
      assert 1 = ops[3][:operand_segment_sizes][0]
      assert 2 = ops[3][:operand_segment_sizes][1]
    end
  end

  test "segment_sizes infer in non-generating usage", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        Func.func _(
                    function_type: Type.function([Type.i32(), Type.i32()], []),
                    sym_name:
                      Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                  ) do
          region do
            block _(a >>> Type.i32(), b >>> Type.i32()) do
              Arith.addi(lhs: a, rhs: b) >>> Type.i32()
              Func.return() >>> []
            end
          end
        end
      end
      |> MLIR.verify!()
    end
    |> Beaver.Composer.append(UseENIFAlloc)
    |> Beaver.Composer.run!()
  end
end
