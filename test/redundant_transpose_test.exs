defmodule RedundantTransposeTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias Beaver.MLIR.{Type}
  alias Beaver.MLIR.Dialect.{Func, TOSA}
  require Func

  test "pass to optimize redundant transpose", context do
    use Beaver
    import Beaver.MLIR.Transforms

    defmodule Helper do
      alias Beaver.MLIR.Attribute
      def perm_t(), do: Type.ranked_tensor([2], Type.i32())

      defp perm_int_attrs() do
        for perm <- 0..1, do: Attribute.integer(Type.i32(), perm)
      end

      def perms_T_attr(), do: Attribute.dense_elements(Enum.reverse(perm_int_attrs()), perm_t())
      def tensor_t(), do: Type.unranked_tensor(Type.f32())
    end

    ir =
      mlir ctx: context[:ctx] do
        module do
          Func.func some_func(
                      function_type: Type.function([Helper.tensor_t()], [Helper.tensor_t()])
                    ) do
            region do
              block bb_entry(arg0 >>> Type.unranked_tensor(Type.f32())) do
                permsT =
                  TOSA.const(value: Helper.perms_T_attr()) >>>
                    Helper.perm_t()

                t = TOSA.transpose(arg0, permsT) >>> Helper.tensor_t()
                t = TOSA.transpose(t, permsT) >>> Helper.tensor_t()
                t = TOSA.transpose(t, permsT) >>> Helper.tensor_t()
                t = TOSA.transpose(t, permsT) >>> Helper.tensor_t()
                Func.return(t) >>> []
              end
            end
          end
        end
        |> MLIR.Operation.verify!(debug: true)
      end

    defmodule DeduplicateTransposePass do
      use Beaver.MLIR.Pass, on: "func.func"

      def const_value(op) do
        operands = Beaver.Walker.operands(op)

        with true <- MLIR.Value.result?(operands[1]),
             const <- MLIR.CAPI.mlirOpResultGetOwner(operands[1]),
             "tosa.const" <- Beaver.MLIR.Operation.name(const) do
          {:ok, Beaver.Walker.attributes(const)["value"]}
        end
      end

      def redundant?(%MLIR.Attribute{} = attr1, %MLIR.Attribute{} = attr2) do
        MLIR.Attribute.equal?(attr1, attr2)
      end

      def run(func) do
        func
        |> Beaver.Walker.prewalk(fn
          x ->
            with %MLIR.Operation{} <- x,
                 "tosa.transpose" <- Beaver.MLIR.Operation.name(x),
                 operands <- Beaver.Walker.operands(x),
                 {:ok, transpose_input_op} <- MLIR.Value.owner(operands[0]),
                 "tosa.transpose" <- Beaver.MLIR.Operation.name(transpose_input_op),
                 {:ok, transpose_perm_attr} <- const_value(x),
                 {:ok, transpose_input_perm_attr} <- const_value(transpose_input_op),
                 true <- redundant?(transpose_perm_attr, transpose_input_perm_attr) do
              Beaver.Walker.replace(x, Beaver.Walker.operands(transpose_input_op)[0])
            else
              _ -> x
            end
        end)

        :ok
      end
    end

    ir_string =
      ir
      |> MLIR.Pass.Composer.nested("func.func", [
        DeduplicateTransposePass.create()
      ])
      |> canonicalize
      |> MLIR.Pass.Composer.run!()
      |> MLIR.to_string()

    assert ir_string =~ "return %arg0 : tensor<*xf32>", ir_string
  end
end
