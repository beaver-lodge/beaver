defmodule RedundantTransposeTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR.{Type}
  alias Beaver.MLIR.Dialect.{Func, TOSA}

  test "pass to optimize redundant transpose", %{ctx: ctx} do
    import Beaver.MLIR.Transform

    ir =
      mlir ctx: ctx do
        function_module some_func(
                          function_type: Type.function([TransposeHelper.tensor_t()], [TransposeHelper.tensor_t()])
                        ) do
          block _bb_entry(arg0 >>> Type.unranked_tensor(Type.f32())) do
            t = TOSA.transpose(arg0, perms: TransposeHelper.perms_t_attr()) >>> TransposeHelper.tensor_t()
            t = TOSA.transpose(t, perms: TransposeHelper.perms_t_attr()) >>> TransposeHelper.tensor_t()
            t = TOSA.transpose(t, perms: TransposeHelper.perms_t_attr()) >>> TransposeHelper.tensor_t()
            t = TOSA.transpose(t, perms: TransposeHelper.perms_t_attr()) >>> TransposeHelper.tensor_t()
            Func.return(t) >>> []
          end
        end
        |> MLIR.verify!()
      end

    ir_string =
      ir
      |> Beaver.Composer.nested(Func.func(), [DeduplicateTransposePass])
      |> canonicalize
      |> Beaver.Composer.run!()
      |> MLIR.to_string()

    assert ir_string =~ "return %arg0 : tensor<*xf32>", ir_string
  end
end
