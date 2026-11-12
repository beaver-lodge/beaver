defmodule Beaver.MLIR.IRBatch.CFG do
  @moduledoc """
  Hygienic syntax over immutable Builder calls. `<-` emits once; `emit` discards a result;
  `step` composes a runtime callback. Block closure, types and dominance remain Builder/MLIR checks.
  """
  alias Beaver.MLIR.IRBatch.Builder

  defmacro cfg(initial, do: body) do
    builder = Macro.unique_var(:builder, __MODULE__)
    forms = statements(body)
    names = for {:block, _, [name | _]} <- forms, is_atom(name) or is_binary(name), do: name
    if length(names) != length(Enum.uniq(names)), do: raise(ArgumentError, "duplicate CFG block")

    quote do
      unquote(builder) = unquote(initial)
      unquote_splicing(Enum.map(forms, &expand(&1, builder)))
      unquote(builder)
    end
  end

  defp expand({:block, meta, [name, [do: body]]}, builder),
    do: expand({:block, meta, [name, [], [do: body]]}, builder)

  defp expand({:block, _, [name, args, [do: body]]}, builder) do
    quote do
      unquote(thread(builder, :start_block, [name, args]))
      unquote_splicing(Enum.map(statements(body), &expand(&1, builder)))
    end
  end

  defp expand({:<-, _, [pattern, value]}, builder),
    do: quote(do: {unquote(pattern), unquote(builder)} = Builder.emit(unquote(builder), unquote(value)))

  defp expand({:emit, meta, [value]}, builder), do: expand({:<-, meta, [{:_, meta, nil}, value]}, builder)

  for {statement, function, arity} <- [{:terminate, :end_block, 1}, {:dispatch, :dispatch, 5}] do
    defp expand({unquote(statement), _, args}, builder) when length(args) == unquote(arity),
      do: thread(builder, unquote(function), args)
  end

  defp expand({:step, _, [callback]}, builder), do: quote(do: unquote(builder) = unquote(callback).(unquote(builder)))

  defp expand({:=, _, _} = assignment, _), do: assignment
  defp expand(other, _), do: raise(ArgumentError, "unsupported CFG statement: #{Macro.to_string(other)}")

  defp thread(builder, function, args) do
    quote do: unquote(builder) = Builder.unquote(function)(unquote(builder), unquote_splicing(args))
  end

  defp statements({:__block__, _, statements}), do: statements
  defp statements(statement), do: [statement]
end
