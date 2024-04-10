defmodule Beaver.MIF.Pass.CreateAbsentFunc do
  use Beaver.MLIR.Pass, on: "func.func"
  use Beaver
  alias Beaver.MLIR.Dialect.Func
  require Func
  import MLIR.CAPI

  defp decompose(call) do
    arg_types =
      Beaver.Walker.operands(call)
      |> Enum.map(&mlirValueGetType/1)

    ret_types =
      Beaver.Walker.results(call)
      |> Enum.map(&mlirValueGetType/1)

    name =
      mlirOperationGetAttributeByName(
        call,
        MLIR.StringRef.create("callee")
      )
      |> mlirSymbolRefAttrGetRootReference()

    {name, arg_types, ret_types}
  end

  def run(func) do
    ctx = mlirOperationGetContext(func)
    block = mlirOperationGetBlock(func)
    symbolTable = mlirSymbolTableCreate(mlirOperationGetParentOperation(func))

    Beaver.Walker.postwalk(
      func,
      fn ir ->
        with op = %MLIR.Operation{} <- ir,
             "func.call" <- MLIR.Operation.name(op),
             {name, arg_types, ret_types} <- decompose(op),
             true <- MLIR.is_null(mlirSymbolTableLookup(symbolTable, name)) do
          mlir ctx: ctx, block: block do
            Func.func _(
                        sym_name: "\"#{MLIR.StringRef.to_string(name)}\"",
                        sym_visibility: MLIR.Attribute.string("private"),
                        function_type: Type.function(arg_types, ret_types)
                      ) do
              region do
              end
            end
          end
        end

        ir
      end
    )

    mlirSymbolTableDestroy(symbolTable)
    :ok
  end
end
