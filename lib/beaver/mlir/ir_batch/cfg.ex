defmodule Beaver.MLIR.IRBatch.CFG do
  @moduledoc """
  A thin macro over the immutable `IRBatch.Builder`.

      cfg Builder.new() do
        block :entry do
          sum <- Builder.expression(:binary, ["addi", [:x, :y], :i64])
          terminate {:return, [sum]}
        end
      end

  `<-` binds an emitted value once; `emit` discards its result. `step callback`
  composes a runtime builder transformation, and `dispatch` delegates to Builder.
  Elixir assignments are allowed for ordinary calculations. The macro only threads
  a hygienic builder variable; MLIR remains responsible for types and dominance.
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
      unquote(builder) = Builder.start_block(unquote(builder), unquote(name), unquote(args))
      unquote_splicing(Enum.map(statements(body), &expand(&1, builder)))
    end
  end

  defp expand({:<-, _, [pattern, value]}, builder) do
    quote do
      {unquote(pattern), unquote(builder)} = Builder.emit(unquote(builder), unquote(value))
    end
  end

  defp expand({:emit, meta, [value]}, builder), do: expand({:<-, meta, [{:_, meta, nil}, value]}, builder)

  defp expand({:terminate, _, [value]}, builder) do
    quote do
      unquote(builder) = Builder.end_block(unquote(builder), unquote(value))
    end
  end

  defp expand({:step, _, [callback]}, builder) do
    quote do
      unquote(builder) = unquote(callback).(unquote(builder))
    end
  end

  defp expand({:dispatch, _, [selector, type, default, cases, emit]}, builder) do
    quote do
      unquote(builder) =
        Builder.dispatch(
          unquote(builder),
          unquote(selector),
          unquote(type),
          unquote(default),
          unquote(cases),
          unquote(emit)
        )
    end
  end

  defp expand({:=, _, _} = assignment, _), do: assignment
  defp expand(other, _), do: raise(ArgumentError, "unsupported CFG statement: #{Macro.to_string(other)}")
  defp statements({:__block__, _, statements}), do: statements
  defp statements(statement), do: [statement]
end
