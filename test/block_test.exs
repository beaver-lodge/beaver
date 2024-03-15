defmodule BlockTest do
  use Beaver.Case, async: true, diagnostic: :server
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.{Func, Arith, CF}
  require Func
  @moduletag :smoke

  test "block usage after defining", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            block do
              v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
              CF.br({Beaver.Env.block(bb1), [v0]}) >>> []
            end

            block bb_a(arg >>> Type.i(32)) do
              Func.return(arg) >>> []
            end

            block _bb_b(arg >>> Type.i(32)) do
              CF.br({Beaver.Env.block(bb_a), [arg]}) >>> []
            end

            block bb1(arg >>> Type.i(32)) do
              v2 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
              add = Arith.addi(arg, v2) >>> Type.i(32)
              Func.return(add) >>> []
            end
          end
        end
      end
    end
    |> MLIR.Operation.verify!()
  end

  test "dangling block", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            block do
              v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
              CF.br({Beaver.Env.block(_bb1), [v0]}) >>> []
            end
          end
        end
      end
    end
    |> MLIR.Operation.verify()

    assert Beaver.Diagnostic.Server.flush(test_context[:diagnostic_server]) =~
             "reference to block defined in another region"
  end

  test "successor of wrong arg type", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            block do
              v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
              CF.br({Beaver.Env.block(bb1), [v0]}) >>> []
            end

            block bb1() do
            end
          end
        end
      end
    end
    |> MLIR.Operation.verify()

    assert Beaver.Diagnostic.Server.flush(test_context[:diagnostic_server]) =~
             "branch has 1 operands for successor"
  end

  test "nested block creation", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        top_level_block = Beaver.Env.block()

        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            %Beaver.MLIR.Block{} =
              block do
                v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                CF.br({Beaver.Env.block(bb1), [v0]}) >>> []
              end

            %Beaver.MLIR.Block{} =
              block bb1() do
                block_1 = Beaver.Env.block()

                %Beaver.MLIR.Block{} =
                  block do
                    refute Beaver.Env.block() == block_1
                    refute Beaver.Env.block() == top_level_block
                  end

                assert Beaver.Env.block() == block_1
              end

            assert Beaver.Env.block() == top_level_block
          end
        end
      end
    end
    |> MLIR.Operation.verify()

    assert Beaver.Diagnostic.Server.flush(test_context[:diagnostic_server]) =~
             "branch has 1 operands for successor"
  end

  test "manual block appending", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          b =
            block do
            end

          region do
            block do
              v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
              Func.return(v0) >>> []
            end

            MLIR.CAPI.mlirRegionAppendOwnedBlock(Beaver.Env.region(), b)
          end
        end
      end
    end
    |> MLIR.Operation.verify()

    assert Beaver.Diagnostic.Server.flush(test_context[:diagnostic_server]) =~
             "empty block: expect at least a terminator"
  end

  test "block in env got popped", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            block do
              v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
              Func.return(v0) >>> []
            end
          end
        end
      end
      |> MLIR.Operation.verify!()

      assert {:not_found, [file: __ENV__.file, line: 145]} == Beaver.Env.block()
    end
  end
end
