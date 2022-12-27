defmodule Beaver.DSL.SSA do
  @moduledoc false
  alias Beaver.MLIR
  require Beaver.MLIR.CAPI

  defstruct arguments: [],
            results: [],
            filler: nil,
            block: nil,
            ctx: nil,
            loc: nil,
            evaluator: nil

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

  def put_filler(%__MODULE__{} = ssa, filler) when is_function(filler, 0) do
    %__MODULE__{ssa | filler: filler}
  end

  def put_block(%__MODULE__{} = ssa, block) do
    %__MODULE__{ssa | block: block}
  end

  def put_ctx(%__MODULE__{} = ssa, %MLIR.CAPI.MlirContext{} = ctx) do
    %__MODULE__{ssa | ctx: ctx}
  end

  # construct ssa by injecting MLIR context and block in the environment
  defp construct_ssa(
         {:>>>, _line, [{_call, line, args}, results]},
         evaluator
       ) do
    quote do
      loc =
        Beaver.MLIR.Location.file(
          name: __ENV__.file,
          line: Keyword.get(unquote(line), :line),
          ctx: MLIR.__CONTEXT__()
        )

      args = List.flatten([unquote_splicing(args)])

      %Beaver.DSL.SSA{evaluator: unquote(evaluator)}
      |> Beaver.DSL.SSA.put_arguments(args)
      |> Beaver.DSL.SSA.put_location(loc)
      |> Beaver.DSL.SSA.put_block(MLIR.__BLOCK__())
      |> Beaver.DSL.SSA.put_ctx(MLIR.__CONTEXT__())
      |> Beaver.DSL.SSA.put_results(unquote(results))
    end
  end

  # block arguments
  defp do_transform(
         {:>>>, _, [var = {_var_name, _, nil}, type]},
         _evaluator
       ) do
    quote do
      {unquote(var), unquote(type)}
    end
  end

  # with do block
  defp do_transform(
         {:>>>, line0, [{call, line, [args, [do: ast_block]]}, results]},
         evaluator
       ) do
    quote do
      unquote(
        construct_ssa(
          {:>>>, line0, [{call, line, args}, results]},
          evaluator
        )
      )
      |> Beaver.DSL.SSA.put_filler(fn -> unquote(ast_block) end)
      |> unquote({call, line, []})
    end
  end

  # op creation
  defp do_transform(
         {:>>>, _line, [{call, line, _args}, _results]} = ast,
         evaluator
       ) do
    quote do
      unquote(construct_ssa(ast, evaluator))
      |> unquote({call, line, []})
    end
  end

  defp do_transform(ast, _evaluator), do: ast

  def prewalk(ast, evaluator) do
    Macro.prewalk(ast, &do_transform(&1, evaluator))
  end

  def postwalk(ast, evaluator) do
    Macro.postwalk(ast, &do_transform(&1, evaluator))
  end
end
