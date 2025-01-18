defmodule BlockHelper do
  @moduledoc false
  use Beaver
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func

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
