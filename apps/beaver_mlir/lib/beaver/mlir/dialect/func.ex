defmodule Beaver.MLIR.Dialect.Func do
  defmacro func(call, do: block) do
    # block |> IO.inspect()
    # call |> IO.inspect(label: "call")
    # mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results) |> mlirTypeAttrGet
    quote do
      unquote(block)
    end
  end
end
