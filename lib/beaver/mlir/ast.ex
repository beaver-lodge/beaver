defmodule Beaver.MLIR.AST do
  @moduledoc """
  Use Elixir AST to express MLIR
  """

  use Beaver

  defp type({t, _line, nil}), do: type(t)

  defp type(t) when is_atom(t) do
    quote do
      MLIR.Type.get(unquote("#{t}"))
    end
  end

  # getting an attribute
  defp argument({k, {:"::", _line0, [v, {t, _line1, nil}]}}) do
    quote do
      {unquote(k), MLIR.Attribute.get(unquote("#{v}: #{t}"))}
    end
  end

  # using a value
  defp argument({{:., _line0, [owner, underlined_n]}, _line1, []}) do
    "_" <> n = underlined_n |> Atom.to_string()
    {i, ""} = Integer.parse(n)

    quote do
      MLIR.CAPI.mlirOperationGetResult(unquote(owner), unquote(i))
    end
  end

  # referencing a block argument, or create a block doesn't exist yet
  defp argument({arg, _line, nil} = ast) when is_atom(arg) do
    quote do
      Beaver.Env.block(unquote(ast))
    end
  end

  defp do_build_ssa(args, ast_result_types, op, clauses \\ []) do
    ast_result_types = Enum.map(ast_result_types, &type/1)
    arguments = Enum.map(args, &argument/1)

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
        loc: Beaver.Env.location(),
        filler: fn ->
          unquote(regions)
        end
      }
      |> Beaver.SSA.put_results([unquote_splicing(ast_result_types)])
      |> MLIR.Operation.create()
    end
  end

  # compile a clause to a block

  defp do_blk(call, expressions) do
    # only support clause with one match call
    [{block_name, _line0, args}] = call

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
        (unquote_splicing(List.wrap(expressions)))
      end
    end
  end

  defp blk([
         [],
         expressions
       ]) do
    do_blk([quote(do: _())], expressions)
  end

  defp blk([
         [{:_, _line0, nil}],
         expressions
       ]) do
    blk([[], expressions])
  end

  defp blk([
         call,
         {:__block__, _line0, expressions}
       ]) do
    do_blk(call, expressions)
  end

  defp blk([
         call,
         {:"::", _line0, _} = expr
       ]) do
    do_blk(call, expr)
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

    if System.get_env("AST_IR_DEBUG") == "1" do
      ast |> Macro.postwalk(&Macro.expand(&1, __CALLER__)) |> Macro.to_string() |> IO.puts()
    end

    quote do
      def unquote(name)(ctx) do
        use Beaver

        mlir ctx: ctx do
          module(unquote(ast))
        end
      end
    end
  end

  defmacro op(
             {:"::", _,
              [
                call,
                types
              ]},
             block \\ []
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
