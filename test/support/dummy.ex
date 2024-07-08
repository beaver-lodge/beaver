defmodule Beaver.Dummy do
  use Beaver
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.{Func, Arith, CF}
  require Func

  defp put_func(ctx, block) do
    mlir ctx: ctx, block: block do
      Func.func some_func(
                  function_type: Type.function([], [Type.i(32)]),
                  sym_name: "\"#{inspect(make_ref())}\""
                ) do
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

  def func_of_3_blocks(ctx) do
    mlir ctx: ctx do
      module do
        put_func(ctx, Beaver.Env.block())
      end
    end
  end

  def readme(ctx) do
    mlir ctx: ctx do
      module do
        unquote(File.read!("test/readme_example.exs") |> Code.string_to_quoted!())
      end
    end
  end

  def gigantic(ctx) do
    mlir ctx: ctx do
      module do
        for _ <- 1..10_000 do
          put_func(ctx, Beaver.Env.block())
        end
      end
    end
  end
end
