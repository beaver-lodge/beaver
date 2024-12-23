defmodule BlockTest do
  use Beaver.Case, async: true, diagnostic: :server
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.{Func, Arith, CF}
  require Func
  @moduletag :smoke

  test "block usage after defining", %{ctx: ctx} do
    mlir ctx: ctx do
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
    |> MLIR.verify!()
  end

  test "dangling block", %{ctx: ctx, diagnostic_server: diagnostic_server} do
    mlir ctx: ctx do
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
    |> MLIR.verify()

    assert Beaver.Capturer.collect(diagnostic_server) =~
             "reference to block defined in another region"
  end

  test "successor of wrong arg type", %{ctx: ctx, diagnostic_server: diagnostic_server} do
    mlir ctx: ctx do
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
    |> MLIR.verify()

    assert Beaver.Capturer.collect(diagnostic_server) =~
             "branch has 1 operands for successor"
  end

  test "nested block creation", %{ctx: ctx, diagnostic_server: diagnostic_server} do
    mlir ctx: ctx do
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
    |> MLIR.verify()

    assert Beaver.Capturer.collect(diagnostic_server) =~
             "branch has 1 operands for successor"
  end

  defmodule BlockHelper do
    def create_ir_by_action(ctx, action) do
      mlir ctx: ctx do
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

              case action do
                :append ->
                  MLIR.Region.append(Beaver.Env.region(), b)

                :insert ->
                  MLIR.Region.insert(Beaver.Env.region(), 1, b)
              end
            end
          end
        end
      end
      |> MLIR.verify()
    end
  end

  @moduledoc false
  describe "insert block to region" do
    for action <- [:insert, :append] do
      test "appending #{action}", %{ctx: ctx, diagnostic_server: diagnostic_server} do
        BlockHelper.create_ir_by_action(ctx, unquote(action))

        assert Beaver.Capturer.collect(diagnostic_server) =~
                 "expect at least a terminator"
      end
    end
  end

  test "block in env got popped", %{ctx: ctx} do
    file = __ENV__.file
    line = __ENV__.line + 3

    mlir ctx: ctx do
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
      |> MLIR.verify!()

      assert {:not_found, [file: ^file, line: ^line]} = Beaver.Env.block()
    end
  end

  test "block arg has no owner", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        Func.func some_func(function_type: Type.function([Type.i(32)], [Type.i(32)])) do
          region do
            block _(a >>> Type.i(32)) do
              assert_raise ArgumentError, "not a result", fn -> MLIR.Value.owner!(a) end

              {const, v0} =
                Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> {:op, Type.i(32)}

              {:ok, owner} = MLIR.Value.owner(v0)
              assert MLIR.equal?(owner, const)
              Func.return(v0) >>> []
            end
          end
        end
      end
      |> MLIR.verify!()
    end
  end

  test "no block to insert to", %{ctx: ctx} do
    assert_raise CompileError, "nofile:1: no valid block in the environment", fn ->
      Code.eval_quoted(
        quote do
          mlir ctx: var!(ctx) do
            Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> {:op, Type.i(32)}
          end
        end,
        ctx: ctx
      )
    end
  end
end
