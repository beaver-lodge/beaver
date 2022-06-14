defmodule Beaver.MLIR.Pattern.CodeGen do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  defmodule Context do
    @enforce_keys [:mlir_ctx, :ex_loc, :block]
    defstruct mlir_ctx: nil,
              ex_loc: nil,
              block: nil,
              named_variables: %{},
              tmp_variables: %{},
              root: nil
  end

  def from_ast(
        ast =
          {:%, [line: line],
           [
             {{:., _, [{dialect, _, _}, op_name]}, _, []},
             {:%{}, _, []}
           ]},
        acc = %Context{
          mlir_ctx: mlir_ctx,
          ex_loc: ex_loc,
          block: block,
          tmp_variables: tmp_variables,
          root: root
        }
      ) do
    loc = MLIR.Location.get!(mlir_ctx, file: ex_loc[:file], line: line)
    full_op_name = Enum.join([dialect, op_name], ".")

    pdl_operation_op =
      MLIR.Dialect.PDL.Operation.State.get(location: loc, name: full_op_name)
      |> MLIR.Operation.create()

    CAPI.mlirBlockAppendOwnedOperation(block, pdl_operation_op)
    pdl_operation_op_result = CAPI.mlirOperationGetResult(pdl_operation_op, 0)
    root = if root, do: root, else: pdl_operation_op_result

    {ast,
     %{
       acc
       | tmp_variables: Map.put(tmp_variables, ast, pdl_operation_op_result),
         root: root
     }}
  end

  def from_ast(
        ast =
          {:=, [line: _line],
           [
             {variable, [line: _line_var], nil},
             binding
           ]},
        acc = %Context{named_variables: named_variables, tmp_variables: tmp_variables}
      ) do
    {found, tmp_variables} = Map.pop!(tmp_variables, binding)

    {ast,
     %{
       acc
       | tmp_variables: tmp_variables,
         named_variables: Map.put(named_variables, variable, found)
     }}
  end

  def from_ast(
        ast = {:erase, [line: line], [{variable, [line: _line], nil}]},
        acc = %Context{
          named_variables: named_variables,
          mlir_ctx: mlir_ctx,
          ex_loc: ex_loc,
          block: block
        }
      ) do
    erasee = Map.fetch!(named_variables, variable)

    loc = MLIR.Location.get!(mlir_ctx, file: ex_loc[:file], line: line)

    pdl_erase_operation_op =
      MLIR.Dialect.PDL.Erase.State.get(location: loc, erasee: erasee)
      |> MLIR.Operation.create()

    CAPI.mlirBlockAppendOwnedOperation(block, pdl_erase_operation_op)
    {ast, acc}
  end

  def from_ast(ast, acc) do
    {ast, acc}
  end
end
