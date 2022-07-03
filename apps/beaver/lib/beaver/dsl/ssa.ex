defmodule Beaver.DSL.SSA do
  alias Beaver.MLIR
  defstruct arguments: [], results: [], filler: nil

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

  def put_filler(%__MODULE__{} = ssa, filler) when is_function(filler) do
    %__MODULE__{ssa | filler: filler}
  end

  defp do_transform(
         {:>>>, _line,
          [
            {call, line, [args, [do: ast_block]]},
            results
          ]}
       ) do
    empty_call = {call, line, [[do: ast_block]]}

    ast =
      quote do
        args = List.flatten([unquote_splicing(args)])

        %Beaver.DSL.SSA{}
        |> Beaver.DSL.SSA.put_arguments(args)
        |> Beaver.DSL.SSA.put_results(unquote(results))
        |> Beaver.DSL.SSA.put_filler(fn -> unquote(ast_block) end)
        |> unquote(empty_call)
      end

    ast
  end

  defp do_transform(
         {:>>>, _line,
          [
            {call, line, args},
            results
          ]}
       ) do
    empty_call = {call, line, []}

    quote do
      args = List.flatten([unquote_splicing(args)])

      %Beaver.DSL.SSA{}
      |> Beaver.DSL.SSA.put_arguments(args)
      |> Beaver.DSL.SSA.put_results(unquote(results))
      |> unquote(empty_call)
    end
  end

  # block arguments
  defp do_transform(
         {:"::", _,
          [
            var = {_var_name, _, nil},
            type
          ]}
       ) do
    quote do
      {unquote(var), unquote(type)}
    end
  end

  defp do_transform(ast), do: ast

  def transform(ast) do
    Macro.prewalk(ast, &do_transform/1)
  end
end
