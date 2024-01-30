defmodule Beaver.MLIR.AST do
  @moduledoc """
  Use Elixir AST to express MLIR
  """

  use Beaver

  defp normalize(ast) do
    ast
  end

  defp type(t) when is_atom(t) do
    quote do
      MLIR.Type.get(unquote("#{t}"))
    end
  end

  defp type({t, _line, nil}) do
    quote do
      MLIR.Type.get(unquote("#{t}"))
    end
  end

  defp argument({k, {:"::", _line0, [v, {t, _line1, nil}]}}) do
    quote do
      {unquote(k), MLIR.Attribute.get(unquote("#{v}: #{t}"))}
    end
  end

  defp argument({{:., _line0, [owner, underlined_n]}, _line1, []}) do
    "_" <> n = underlined_n |> Atom.to_string()
    {i, ""} = Integer.parse(n)

    quote do
      MLIR.CAPI.mlirOperationGetResult(unquote(owner), unquote(i))
    end
  end

  defp argument({arg, _line, nil} = ast) when is_atom(arg) do
    ast
  end

  defp do_build_ssa(args, ast_result_types, op, clauses \\ []) do
    ast_result_types = Enum.map(ast_result_types, &type/1)
    arguments = Enum.map(args, &argument/1)
    clauses = clauses

    regions =
      case clauses do
        [] ->
          []

        clauses when is_list(clauses) ->
          quote do
            region do
              (unquote_splicing(clauses))
            end
          end
      end

    quote do
      %SSA{
        arguments: unquote(arguments),
        ctx: Beaver.Env.context(),
        block: Beaver.Env.block(),
        filler: fn ->
          unquote(regions)
        end
      }
      |> SSA.put_results([unquote_splicing(ast_result_types)])
      |> then(&MLIR.Operation.create(%Beaver.SSA{&1 | op: unquote("#{op}")}))
    end
  end

  defp build_ssa(
         {{:., _line0, [Access, :get]}, _line1,
          [
            {{:., _line2, [{:__aliases__, _line3, [:MLIR]}, op]}, _line4, args},
            attributes
          ]},
         ast_result_types
       ) do
    do_build_ssa(args ++ attributes, ast_result_types, op)
  end

  defp build_ssa(
         {{:., _line0, [{:__aliases__, _line1, [:MLIR]}, op]}, _line2, args},
         ast_result_types
       ) do
    do_build_ssa(args, ast_result_types, op)
  end

  defp build_ssa(
         {{{:., _line0, [{:__aliases__, _line1, [:MLIR]}, op]}, _line2, args}, _line3,
          attributes},
         ast_result_types,
         clauses \\ []
       ) do
    do_build_ssa(args ++ attributes, ast_result_types, op, clauses)
  end

  # compile a clause to a block

  defp blk_jump({block_name, _line0, args})
       when is_atom(block_name) and :"::" != block_name do
    args = args |> Enum.map(&argument/1)

    quote do
      MLIR.Dialect.CF.br(
        {Beaver.Env.block(unquote(Macro.var(block_name, nil))), [unquote_splicing(args)]}
      ) >>>
        []
    end
  end

  defp blk_jump(expr) do
    expr
  end

  defp do_blk(args, expressions) do
    expressions = expressions |> Enum.map(&blk_jump/1)

    {block_name, args} =
      case args do
        [
          {block_name, _line0, args}
        ] ->
          {block_name, args}

        [] ->
          {:__anonymous__, []}
      end

    types =
      for {:"::", _line0, [{_arg, _line1, nil}, t]} <- args do
        type(t)
      end

    vars_of_blk_args =
      for {{:"::", _line0, [var, _t]}, index} <- Enum.with_index(args) do
        quote do
          Kernel.var!(unquote(var)) = MLIR.Block.get_arg!(Beaver.Env.block(), unquote(index))
        end
      end

    quote do
      block unquote(block_name)() do
        MLIR.Block.add_arg!(Beaver.Env.block(), Beaver.Env.context(), [unquote_splicing(types)])
        unquote_splicing(vars_of_blk_args)
        (unquote_splicing(expressions))
      end
    end
  end

  defp blk([
         [{:_, _line0, nil}],
         {:__block__, _line1, expressions}
       ]) do
    do_blk([], expressions)
  end

  defp blk([
         args,
         {:__block__, _line0, expressions}
       ]) do
    do_blk(args, expressions)
  end

  defp blk([
         args,
         {:"::", _line0, _} = expr
       ]) do
    do_blk(args, [expr])
  end

  defp to_ssa({:"::", _line0, [{:=, line1, [var | [bind]]} | [{:{}, _, result_types}]]}) do
    bind =
      quote do
        unquote(build_ssa(bind, result_types))
      end

    {:=, line1, [var | [bind]]}
  end

  defp to_ssa({:"::", _line0, [expr | [{:{}, _, result_types}]]}) do
    quote do
      unquote(build_ssa(expr, result_types))
    end
  end

  defp to_ssa(
         {:"::", _line0,
          [
            {{:., _line6, [Access, :get]}, _line5,
             [
               expr =
                 {{:., _line1,
                   [
                     {:__aliases__, line2, [:MLIR]},
                     op
                   ]}, _line3, []},
               attributes
             ]},
            {result_type, _line4,
             [
               [
                 do: do_block
               ]
             ]}
          ]}
       )
       when is_atom(op) do
    result_types = [result_type]

    clauses =
      for {:->, _line, clause} <- do_block do
        blk(clause)
      end

    quote do
      unquote(build_ssa({expr, line2, attributes}, result_types, clauses))
    end
  end

  defp to_ssa(ast) do
    ast
  end

  defmacro defm(ast) do
    ast =
      ast[:do]
      |> normalize
      |> Macro.prewalk(&to_ssa/1)

    ast |> Macro.to_string() |> IO.puts()

    quote do
      use Beaver
      alias Beaver.SSA
      ctx = MLIR.Context.create(allow_unregistered: true)

      mlir ctx: ctx do
        module do
          unquote(ast)
        end
      end
    end
  end
end
