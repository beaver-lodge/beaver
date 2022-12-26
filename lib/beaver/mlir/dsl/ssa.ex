defmodule Beaver.DSL.SSA do
  alias Beaver.MLIR
  require Beaver.MLIR.CAPI
  defstruct arguments: [], results: [], filler: nil, block: nil, ctx: nil, loc: nil

  def put_arguments(%__MODULE__{arguments: arguments} = ssa, additional_arguments)
      when is_list(additional_arguments) do
    %__MODULE__{ssa | arguments: arguments ++ additional_arguments}
  end

  def put_results(%__MODULE__{results: results} = ssa, f) when is_function(f, 1) do
    %__MODULE__{ssa | results: results ++ [f]}
  end

  def put_results(%__MODULE__{results: results} = ssa, %MLIR.CAPI.MlirType{} = single_result) do
    %__MODULE__{ssa | results: results ++ [single_result]}
  end

  def put_results(%__MODULE__{results: results} = ssa, additional_results)
      when is_list(additional_results) do
    %__MODULE__{ssa | results: results ++ additional_results}
  end

  def put_location(%__MODULE__{} = ssa, %MLIR.CAPI.MlirLocation{} = loc) do
    %__MODULE__{ssa | loc: loc}
  end

  def put_filler(%__MODULE__{} = ssa, filler) when is_function(filler) do
    %__MODULE__{ssa | filler: filler}
  end

  def put_block(%__MODULE__{} = ssa, block) do
    %__MODULE__{ssa | block: block}
  end

  def put_ctx(%__MODULE__{} = ssa, %MLIR.CAPI.MlirContext{} = ctx) do
    %__MODULE__{ssa | ctx: ctx}
  end

  # block arguments
  defp do_transform(
         {:>>>, _,
          [
            var = {_var_name, _, nil},
            type
          ]}
       ) do
    quote do
      {unquote(var), unquote(type)}
    end
  end

  # with do block
  defp do_transform(
         {:>>>, _line,
          [
            {call, line, [args, [do: ast_block]]},
            results
          ]}
       ) do
    empty_call = {call, line, []}

    ast =
      quote do
        loc =
          Beaver.MLIR.Location.file(
            name: __ENV__.file,
            line: Keyword.get(unquote(line), :line),
            ctx: MLIR.__CONTEXT__()
          )

        args = List.flatten([unquote_splicing(args)])

        %Beaver.DSL.SSA{}
        |> Beaver.DSL.SSA.put_filler(fn -> unquote(ast_block) end)
        |> Beaver.DSL.SSA.put_arguments(args)
        |> Beaver.DSL.SSA.put_location(loc)
        |> Beaver.DSL.SSA.put_block(MLIR.__BLOCK__())
        |> Beaver.DSL.SSA.put_ctx(MLIR.__CONTEXT__())
        |> Beaver.DSL.SSA.put_results(unquote(results))
        |> unquote(empty_call)
      end

    ast
  end

  # op creation
  defp do_transform(
         {:>>>, _line,
          [
            {call, line, args},
            results
          ]}
       ) do
    empty_call = {call, line, []}

    quote do
      loc =
        Beaver.MLIR.Location.file(
          name: __ENV__.file,
          line: Keyword.get(unquote(line), :line),
          ctx: MLIR.__CONTEXT__()
        )

      args = List.flatten([unquote_splicing(args)])

      %Beaver.DSL.SSA{}
      |> Beaver.DSL.SSA.put_arguments(args)
      |> Beaver.DSL.SSA.put_location(loc)
      |> Beaver.DSL.SSA.put_block(MLIR.__BLOCK__())
      |> Beaver.DSL.SSA.put_ctx(MLIR.__CONTEXT__())
      |> Beaver.DSL.SSA.put_results(unquote(results))
      |> unquote(empty_call)
    end
  end

  defp do_transform(ast), do: ast

  def transform(ast) do
    Macro.prewalk(ast, &do_transform/1)
  end
end
