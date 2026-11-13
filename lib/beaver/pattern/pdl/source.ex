defmodule Beaver.Pattern.PDL.Source do
  @moduledoc "PDL text constructors. Names and literal attribute/type fragments are trusted MLIR syntax; parsing verifies the result."
  def declarations(operands, types, attributes) do
    [
      Enum.map(operands, &"%#{&1} = pdl.operand\n"),
      Enum.map(types, fn {name, type} -> "%#{name} = pdl.type#{if type, do: " : #{type}"}\n" end),
      Enum.map(attributes, fn {name, value} -> "%#{name} = pdl.attribute = #{value}\n" end)
    ]
  end

  def typed(names, kind),
    do: Enum.map_join(names, ", ", &"%#{&1}") <> " : " <> Enum.map_join(names, ", ", fn _ -> "!pdl.#{kind}" end)

  def result(name, operation), do: "%#{name} = pdl.result 0 of %#{operation}\n"
  def replace(root, operation), do: "pdl.replace %#{root} with %#{operation}\n"
  def replace_values(root, values), do: "pdl.replace %#{root} with (#{typed(values, "value")})\n"
  def call(name, symbol, args, results), do: operation(name, "func.call", args, results, callee: symbol)

  def value(name, op, args, results, attributes \\ []) do
    operation = "#{name}_op"
    [operation(operation, op, args, results, attributes), result(name, operation)]
  end

  def operation(name, op, args, results, attributes) do
    operands = if args == [], do: "", else: "(#{typed(args, "value")})"

    attrs =
      if attributes == [],
        do: "",
        else: " {" <> Enum.map_join(attributes, ", ", fn {key, value} -> ~s("#{key}" = %#{value}) end) <> "}"

    "%#{name} = pdl.operation \"#{op}\"#{operands}#{attrs} -> (#{typed(results, "type")})\n"
  end

  def rule(name, callee, benefit, match, root, rewrite),
    do:
      {callee,
       IO.iodata_to_binary([
         "pdl.pattern @#{name} : benefit(#{benefit}) {\n",
         match,
         "pdl.rewrite %#{root} {\n",
         rewrite,
         "}\n}\n"
       ])}

  @doc """
  Builds a rule from `{results, operation, operands, types, attributes}` steps and replacement values.
  Result/operand names are symbolic; string types are concrete and atom types are shared wildcards.
  The last match operation is the root. Constants, types and free operands are declared once.
  """
  def pattern(name, callee, benefit, match, rewrite, replacements) when match != [] do
    steps = match ++ rewrite
    types = Enum.flat_map(steps, &elem(&1, 3)) |> Enum.uniq() |> Enum.with_index(&{&1, "t#{&2}"})
    attrs = Enum.flat_map(steps, &Keyword.values(elem(&1, 4))) |> Enum.uniq() |> Enum.with_index(&{&1, "a#{&2}"})
    bound = Enum.flat_map(match, &List.wrap(elem(&1, 0)))
    operands = Enum.flat_map(match, &elem(&1, 2)) |> Enum.uniq() |> Enum.reject(&(&1 in bound))

    declarations =
      declarations(
        operands,
        Enum.map(types, fn {t, n} -> {n, if(is_binary(t), do: t)} end),
        Enum.map(attrs, fn {v, n} -> {n, v} end)
      )

    render = &pattern_steps(&1, &2, Map.new(types), Map.new(attrs))
    root = "match#{length(match) - 1}"

    rule(name, callee, benefit, [declarations, render.(match, "match")], root, [
      render.(rewrite, "rewrite"),
      replace_values(root, replacements)
    ])
  end

  defp pattern_steps(steps, prefix, types, attrs) do
    Enum.with_index(steps, fn {results, op, operands, result_types, attributes}, i ->
      results = List.wrap(results)
      unless length(results) == length(result_types), do: raise(ArgumentError, "PDL result/type count mismatch")
      name = "#{prefix}#{i}"
      attributes = Enum.map(attributes, fn {k, v} -> {k, Map.fetch!(attrs, v)} end)

      [
        operation(name, op, operands, Enum.map(result_types, &Map.fetch!(types, &1)), attributes),
        Enum.with_index(results, fn result, index -> "%#{result} = pdl.result #{index} of %#{name}\n" end)
      ]
    end)
  end

  @doc "Retargets a call with any operand list and explicit result types; returns a rule entry for PDL.new/1."
  def retarget(name, from, to, results, benefit \\ 1) do
    types = Enum.with_index(results, &{"t#{&2}", &1})

    call = fn name, symbol ->
      "%#{name} = pdl.operation \"func.call\"(%args : !pdl.range<value>) {\"callee\" = %#{symbol}} -> (#{typed(Enum.map(types, &elem(&1, 0)), "type")})\n"
    end

    match = [
      "%args = pdl.operands\n",
      declarations([], types,
        src: Beaver.MLIR.Template.check_symbol!(from),
        dst: Beaver.MLIR.Template.check_symbol!(to)
      ),
      call.(:call, :src)
    ]

    rule(name, from, benefit, match, :call, [call.(:bound, :dst), replace(:call, :bound)])
  end
end
