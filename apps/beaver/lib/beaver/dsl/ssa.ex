defmodule Beaver.DSL.SSA do
  alias Beaver.MLIR
  defstruct arguments: [], results: []

  def put_arguments(%__MODULE__{arguments: arguments} = ssa, additional_arguments)
      when is_list(additional_arguments) do
    %__MODULE__{ssa | arguments: arguments ++ additional_arguments}
  end

  def put_results(%__MODULE__{results: results} = ssa, %MLIR.CAPI.MlirType{} = single_result) do
    %__MODULE__{ssa | results: results ++ [single_result]}
  end

  def put_results(%__MODULE__{results: results} = ssa, additional_results)
      when is_list(additional_results) do
    %__MODULE__{ssa | results: results ++ additional_results}
  end

  defp do_transform(
         {:>>>, _line,
          [
            {call, line, args},
            results
          ]}
       ) do
    empty_call = {call, line, []}

    ast =
      quote do
        args = List.flatten([unquote_splicing(args)])

        %Beaver.DSL.SSA{}
        |> Beaver.DSL.SSA.put_arguments(args)
        |> Beaver.DSL.SSA.put_results(unquote(results))
        |> unquote(empty_call)
      end

    ast
  end

  defp do_transform(ast), do: ast

  def transform(ast) do
    Macro.prewalk(ast, &do_transform/1)
  end
end
