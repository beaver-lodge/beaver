defmodule RedundantTransposeTest do
  use ExUnit.Case

  alias Beaver.MLIR
  alias Beaver.MLIR.{Type, Attribute, ODS}
  alias Beaver.MLIR.Dialect.{Func, TOSA}

  test "pass to optimize redundant transpose" do
    use Beaver
    import Beaver.MLIR.Transforms

    defmodule Helper do
      def perm_t(), do: Type.ranked_tensor([2], Type.i32())

      defp perm_int_attrs() do
        for perm <- [0, 1] do
          Attribute.integer(Type.i32(), perm)
        end
      end

      def perms_attr(), do: Attribute.dense_elements(perm_int_attrs(), perm_t())
      def perms_T_attr(), do: Attribute.dense_elements(Enum.reverse(perm_int_attrs()), perm_t)
      def tensor_t(), do: Type.unranked_tensor(Type.f32())
    end

    ir =
      mlir do
        module do
          Func.func some_func(
                      function_type: Type.function([Helper.tensor_t()], [Helper.tensor_t()])
                    ) do
            region do
              block bb_entry(arg0 >>> Type.unranked_tensor(Type.f32())) do
                perms = TOSA.const(value: Helper.perms_attr()) >>> Helper.perm_t()

                perms1 =
                  TOSA.const(value: Helper.perms_T_attr()) >>>
                    Helper.perm_t()

                t = TOSA.transpose(arg0, perms) >>> Helper.tensor_t()
                t = TOSA.transpose(t, perms1) >>> Helper.tensor_t()
                t = TOSA.transpose(t, perms) >>> Helper.tensor_t()
                _t = TOSA.transpose(t, perms1) >>> Helper.tensor_t()
                Func.return(t) >>> []
              end
            end
          end
        end
        |> MLIR.Operation.verify!(dump_if_fail: true)
        |> MLIR.Operation.from_module()
        |> MLIR.Operation.dump!()
      end

    defmodule DeduplicateTransposePass do
      use Beaver.MLIR.Pass, on: Func.Func

      def const_value(%TOSA.Transpose{operands: operands}) do
        with true <- MLIR.Value.result?(operands[1]),
             const <- MLIR.CAPI.mlirOpResultGetOwner(operands[1]),
             %TOSA.Const{attributes: const_attributes} <-
               Beaver.concrete(const) do
          {:ok, const_attributes["value"]}
        end
      end

      def redundant?(%MLIR.CAPI.MlirAttribute{} = attr1, %MLIR.CAPI.MlirAttribute{} = attr2) do
        case1 =
          MLIR.Attribute.equal?(Helper.perms_attr(), attr1) &&
            MLIR.Attribute.equal?(Helper.perms_T_attr(), attr2)

        case2 =
          MLIR.Attribute.equal?(Helper.perms_attr(), attr2) &&
            MLIR.Attribute.equal?(Helper.perms_T_attr(), attr1)

        case1 || case2
      end

      def run(%MLIR.CAPI.MlirOperation{} = operation) do
        %Func.Func{} = func = Beaver.concrete(operation)

        func
        |> Beaver.Walker.postwalk(fn
          %MLIR.CAPI.MlirOperation{} = operation ->
            with %TOSA.Transpose{operands: operands} = transpose_op <- Beaver.concrete(operation),
                 true <- MLIR.Value.result?(operands[0]),
                 transpose_input_op <- MLIR.CAPI.mlirOpResultGetOwner(operands[0]),
                 %TOSA.Transpose{} = transpose_input_op <-
                   Beaver.concrete(transpose_input_op),
                 {:ok, transpose_perm_attr} <- const_value(transpose_op),
                 {:ok, transpose_input_perm_attr} <- const_value(transpose_input_op),
                 true <- redundant?(transpose_perm_attr, transpose_input_perm_attr) do
              MLIR.Operation.dump!(operation)
              operation
            else
              _ -> operation
            end

          x ->
            x
        end)

        :ok
      end
    end

    ir
    |> MLIR.Pass.Composer.nested(Func.Func, [
      DeduplicateTransposePass.create()
    ])
    |> canonicalize
    |> MLIR.Pass.Composer.run!()
  end
end
