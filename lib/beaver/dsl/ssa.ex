defmodule Beaver.SSA do
  @moduledoc """
  Storing MLIR IR structure with a Elixir struct. Macros like `Beaver.mlir/1` will generate SSA structs defined by this module.
  """
  alias Beaver.MLIR
  require Beaver.MLIR.CAPI

  @type t() :: %__MODULE__{
          arguments: any(),
          results: any(),
          filler: any(),
          block: nil,
          ctx: any(),
          loc: any(),
          evaluator: function()
        }
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

  def put_results(%__MODULE__{results: results} = ssa, %MLIR.Type{} = single_result) do
    %__MODULE__{ssa | results: results ++ [single_result]}
  end

  def put_results(%__MODULE__{results: results} = ssa, additional_results)
      when is_list(additional_results) do
    %__MODULE__{ssa | results: results ++ additional_results}
  end

  def put_results(%__MODULE__{} = ssa, :infer = infer) do
    %__MODULE__{ssa | results: infer}
  end

  def put_location(%__MODULE__{} = ssa, %MLIR.Location{} = loc) do
    %__MODULE__{ssa | loc: loc}
  end

  def put_filler(%__MODULE__{} = ssa, filler) when is_function(filler, 0) do
    %__MODULE__{ssa | filler: filler}
  end

  def put_block(%__MODULE__{} = ssa, block) do
    %__MODULE__{ssa | block: block}
  end

  def put_ctx(%__MODULE__{} = ssa, %MLIR.Context{} = ctx) do
    %__MODULE__{ssa | ctx: ctx}
  end

  # construct ssa by injecting MLIR context and block in the environment
  defp construct_ssa(
         {:>>>, _line, [{_call, line, args}, results]},
         evaluator
       ) do
    quote do
      args = [unquote_splicing(args)] |> List.flatten()
      results = unquote(results) |> List.wrap() |> List.flatten()

      {loc, args} =
        Enum.reduce(args, {nil, []}, fn arg, {loc, args} ->
          case arg do
            %MLIR.Location{} = loc ->
              {loc, args}

            a ->
              {loc, args ++ [a]}
          end
        end)

      loc =
        loc ||
          Beaver.MLIR.Location.file(
            name: __ENV__.file,
            line: Keyword.get(unquote(line), :line),
            ctx: Beaver.Env.context()
          )

      %Beaver.SSA{evaluator: unquote(evaluator)}
      |> Beaver.SSA.put_location(loc)
      |> Beaver.SSA.put_arguments(args)
      |> Beaver.SSA.put_results(results)
      |> Beaver.SSA.put_block(Beaver.Env.block())
      |> Beaver.SSA.put_ctx(Beaver.Env.context())
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
      |> Beaver.SSA.put_filler(fn -> unquote(ast_block) end)
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

  def prewalk(ast, evaluator) when is_function(evaluator, 2) do
    Macro.prewalk(ast, &do_transform(&1, evaluator))
  end

  def postwalk(ast, evaluator) when is_function(evaluator, 2) do
    Macro.postwalk(ast, &do_transform(&1, evaluator))
  end
end
