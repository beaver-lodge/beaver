defmodule Beaver.MLIR.Dialect.Ex.Schema do
  @moduledoc false
  @pure ~w(lit add sub mul cmp)a

  defmacro defcops(layer) do
    alias Beaver.MLIR.Conversion.Ex.CRuntime

    definitions =
      for name <- CRuntime.functions(layer), mode <- [:btt, :value] do
        {inputs, outputs} =
          if mode == :btt, do: {CRuntime.argument_types(name), ["i32"]}, else: CRuntime.value_signature(name)

        args =
          inputs
          |> Enum.with_index()
          |> Enum.map(fn {type, i} ->
            quote do: unquote(Macro.var(String.to_atom("a#{i}"), nil)) = unquote(constraint(type))
          end)

        results = for {type, i} <- Enum.with_index(outputs), do: {String.to_atom("out#{i}"), constraint(type)}
        call = {String.to_atom("#{mode}_" <> name), [], args}
        quote do: Beaver.MLIR.Dialect.Ex.Schema.defop(unquote(call), results: unquote(results))
      end

    {:__block__, [], definitions}
  end

  defp constraint("!ex." <> name), do: quote(do: base(unquote({String.to_atom(name), [], []})))
  defp constraint(type), do: quote(do: Beaver.MLIR.Type.get(unquote(type)))

  def pure_effects(_op), do: :pure
  def speculatable(_op), do: :speculatable

  defmacro defops(names, operands, opts) do
    definitions =
      for name <- Macro.expand(names, __CALLER__) do
        quote do: unquote(__MODULE__).defop(unquote({name, [], operands}), unquote(opts))
      end

    {:__block__, [], definitions}
  end

  defmacro defop(call, opts \\ []) do
    # Fragment variables name schema operands and constraints in the facade.
    {call, opts} =
      Macro.prewalk({call, opts}, fn
        {name, meta, context} when is_atom(name) and is_atom(context) -> {name, meta, nil}
        ast -> ast
      end)

    {name, _} = Macro.decompose_call(call)

    opts =
      if name in @pure do
        Keyword.put(opts, :interfaces,
          memory_effects: quote(do: &unquote(__MODULE__).pure_effects/1),
          conditionally_speculatable: quote(do: &unquote(__MODULE__).speculatable/1)
        )
      else
        opts
      end

    quote do
      @ex_operations {unquote("ex.#{name}"), @ex_layer}
      Beaver.Slang.defop(unquote(call), unquote(opts))
    end
  end
end
