defmodule Beaver.MLIR.Dialect.Func do
  alias Beaver.MLIR

  defmacro func(call, do: block) do
    {func_name, args} = call |> Macro.decompose_call()
    if not is_atom(func_name), do: raise("func name must be an atom")

    func_ast =
      quote do
        # TODO: support getting ctx from opts
        ctx = Beaver.MLIR.Managed.Context.get()

        # create function

        operation_state = ctx |> Beaver.MLIR.Operation.State.get!("func.func")

        inputs = []

        results =
          case unquote(args)[:return_type] do
            nil ->
              []

            return_types when is_list(return_types) ->
              return_types

            return_type ->
              [return_type]
          end

        __func_type__ =
          Beaver.MLIR.CAPI.mlirFunctionTypeGet(
            ctx,
            length(inputs),
            inputs |> Exotic.Value.Array.get() |> Exotic.Value.get_ptr(),
            length(results),
            results |> Exotic.Value.Array.get() |> Exotic.Value.get_ptr()
          )
          |> Beaver.MLIR.CAPI.mlirTypeAttrGet()

        __func_type__ = ~a{() -> i32}

        operation_state =
          Beaver.MLIR.Operation.State.add_attr(operation_state,
            function_type: __func_type__,
            sym_name: "\"#{unquote(func_name)}\""
          )

        unquote(block)

        func_region = Beaver.MLIR.Managed.Region.get()
        Beaver.MLIR.Operation.State.add_regions(operation_state, [func_region])
        func_op = operation_state |> Beaver.MLIR.Operation.create()

        # insert func op into container if avaliabe
        Beaver.MLIR.Managed.InsertionPoint.get().(func_op)
      end

    func_ast |> Macro.to_string()
    func_ast
  end

  def return(arguments) when is_list(arguments) do
    MLIR.Operation.create("func.return", arguments)
  end

  def return(arg) do
    MLIR.Operation.create("func.return", [arg])
  end
end
