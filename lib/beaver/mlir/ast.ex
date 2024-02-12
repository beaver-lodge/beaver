defmodule Beaver.MLIR.AST do
  @moduledoc """
  Use Elixir AST to express MLIR
  """

  use Beaver

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
      %Beaver.SSA{
        op: unquote("#{op}"),
        arguments: unquote(arguments),
        ctx: Beaver.Env.context(),
        block: Beaver.Env.block(),
        filler: fn ->
          unquote(regions)
        end
      }
      |> Beaver.SSA.put_results([unquote_splicing(ast_result_types)])
      |> MLIR.Operation.create()
    end
  end

  # compile a clause to a block

  defp do_blk(args, expressions) do
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
         [{:_, _line0, nil}],
         expressions
       ]) do
    do_blk([], List.wrap(expressions))
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

  defp blk(ast) do
    raise """
    can't process clause body:
    #{Macro.to_string(ast)}
    ast:
    #{inspect(ast, pretty: true)}
    """
  end

  defmacro defm(call, ast) do
    {name, _args} = Macro.decompose_call(call)

    quote do
      def unquote(name)(ctx) do
        use Beaver
        alias Beaver.SSA

        mlir ctx: ctx do
          module(unquote(ast))
        end
      end
    end
  end

  defmacro br(call) do
    {block_name, args} = Macro.decompose_call(call)
    args = args |> Enum.map(&argument/1)

    quote do
      mlir do
        MLIR.Dialect.CF.br(
          {Beaver.Env.block(unquote(Macro.var(block_name, nil))), [unquote_splicing(args)]}
        ) >>>
          []
      end
    end
  end

  defmacro op(ast, block \\ [])

  defmacro op(
             {:"::", _,
              [
                call,
                types
              ]},
             block
           ) do
    {call, attributes} =
      case call do
        {{:., _, [Access, :get]}, _, [call, attributes]} ->
          {call, attributes}

        _ ->
          {call, []}
      end

    {{dialect, _, nil}, op, operands} = Macro.decompose_call(call)
    name = "#{dialect}.#{op}"

    quote do
      op(unquote(name), unquote(operands), unquote(attributes), unquote(types), unquote(block))
    end
  end

  defmacro op(name, operands, attributes, types, block \\ []) do
    types =
      case types do
        {val, _, nil} = t when is_atom(val) ->
          List.wrap(t)

        {:{}, _, types} ->
          types

        _ ->
          Tuple.to_list(types)
      end

    clauses =
      for {:->, _line, clause} <- block[:do] || [] do
        blk(clause)
      end

    do_build_ssa(operands ++ attributes, types, name, clauses)
  end
end
