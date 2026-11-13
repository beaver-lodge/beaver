defmodule Beaver.MLIR.Template.Registers.Patterns do
  @moduledoc "PDL specializations for the tagged register template family, parameterized by symbols, tags and lowering policy."
  alias Beaver.Pattern.PDL
  import PDL.Source

  defmacro __using__(opts) do
    quote do
      @register_patterns Map.new(unquote(opts))
      defp rules, do: unquote(__MODULE__).rules(@register_patterns, __MODULE__)

      use Beaver.Pattern.PDL.Pipeline,
        templates: @register_patterns.templates,
        runtime: @register_patterns.runtime,
        policy: @register_patterns.policy,
        rules: &rules/0
    end
  end

  def rules(config, owner) do
    revision = {config, Enum.map([owner, __MODULE__, config.templates, PDL.Source], & &1.__info__(:md5))}

    PDL.fetch(owner, revision, fn ->
      arithmetic = config.templates.arithmetic_specs()

      config =
        Map.merge(config, %{
          scalar_arith: arithmetic,
          widened_ops: for({name, op} <- arithmetic, op in ~w(addi subi muli), do: name),
          cmp_preds:
            for(
              {name, pred} <- config.templates.comparison_specs(),
              do: {name, Enum.find_index(~w(eq ne slt sle sgt sge), &(&1 == pred))}
            )
        })

      base_entries(config) ++ widened_pdl(config) ++ scalar_arith_pdl(config) ++ cmp_pdl(config)
    end)
  end

  defp c(config, results, symbol, operands, types \\ [:term]),
    do: {results, "func.call", operands, types, [callee: "@" <> config.prefix <> symbol]}

  defp v(result, op, operands, type, attrs \\ []), do: {result, "arith." <> op, operands, [type], attrs}
  defp constant(result, value, type), do: v(result, "constant", [], type, value: value)

  defp boxed(config, symbol, results),
    do: [
      c(config, :xbox, "box_int", [:x]),
      c(config, :ybox, "box_int", [:y]),
      c(config, results, symbol, [:xbox, :ybox], types(results))
    ]

  defp types(results), do: if(is_list(results), do: [:term, "i1"], else: [:term])

  defp spec(config, name, symbol, match, rewrite, results, benefit \\ 3),
    do: pattern(name, config.prefix <> symbol, benefit, match, rewrite, List.wrap(results))

  defp base_entries(config) do
    fields =
      for {field, operand} <- [{"bits", :b}, {"word0", :t}, {"aux", :a}],
          do: {"term_#{field}", "box", [:t, :b, :a], operand}

    fields = fields ++ for(kind <- ~w(int bool), do: {"term_bits", "box_#{kind}", [:b], :b})

    fields =
      for {name, box, args, out} <- fields,
          do:
            spec(
              config,
              "fold_#{name}_of_#{box}",
              "#{name}",
              [c(config, :box, "#{box}", args), c(config, :root, "#{name}", [:box], ["i64"])],
              [],
              out,
              4
            )

    constants =
      for {kind, tag} <- [{"int", config.tags.integer}, {"bool", config.tags.boolean}, {"nil", config.tags.nil}],
          do: constant_fold(config, kind, tag, false)

    truth = for {kind, value} <- [{"int", 1}, {"nil", 0}], do: constant_fold(config, kind, value, true)
    fields ++ constants ++ truth ++ [truth_rule(config), cursor_rule(config), poll_rule(config), predicate_rule(config)]
  end

  defp constant_fold(config, kind, value, truth?) do
    name = if truth?, do: "truth", else: "term_tag"
    symbol = "#{name}"

    match = [
      c(config, :box, "box_#{kind}", if(kind == "nil", do: [], else: [:x])),
      c(config, :root, symbol, [:box], [if(truth?, do: :term, else: "i64")])
    ]

    rewrite =
      [constant(:value, "#{value} : i64", "i64")] ++ if(truth?, do: [c(config, :out, "box_bool", [:value])], else: [])

    spec(
      config,
      "fold_#{name}_of_box_#{kind}",
      symbol,
      match,
      rewrite,
      if(truth?, do: :out, else: :value),
      if(truth?, do: 3, else: 4)
    )
  end

  defp truth_rule(config),
    do:
      spec(
        config,
        "fold_truth_of_box_bool_i1",
        "truth",
        [c(config, :box, "box_bool_i1", [:c]), c(config, :root, "truth", [:box])],
        [],
        :box
      )

  defp cursor_rule(config) do
    match = [c(config, :first, "binary_cursor", [:bin, :d1]), c(config, :root, "binary_cursor", [:first, :d2])]

    spec(
      config,
      "coalesce_binary_cursor",
      "binary_cursor",
      match,
      [v(:sum, "addi", [:d1, :d2], "i64"), c(config, :out, "binary_cursor", [:bin, :sum])],
      :out
    )
  end

  defp poll_rule(config) do
    match = [constant(:zero, "0 : i64", "i64"), c(config, :root, "poll", [:zero], ["i1"])]
    spec(config, "fold_zero_cost_poll", "poll", match, [constant(:out, "false", "i1")], :out, 4)
  end

  defp predicate_rule(config) do
    match = [
      c(config, :box, "box_int", [:x]),
      c(config, [:root, :valid], "try_predicate", [:box, :expected], [:term, "i1"])
    ]

    rewrite = [
      constant(:one, "#{config.tags.integer} : i64", "i64"),
      v(:cmp, "cmpi", [:one, :expected], "i1", predicate: "0 : i64"),
      c(config, :out, "box_bool_i1", [:cmp]),
      constant(:ok, "true", "i1")
    ]

    spec(config, "spec_try_predicate_known_int", "try_predicate", match, rewrite, [:out, :ok])
  end

  defp binary_spec(config, name, checked?, rewrite) do
    roots = if checked?, do: [:root, :valid], else: :root

    spec(
      config,
      "spec_#{name}_known_i64",
      name,
      boxed(config, name, roots),
      rewrite,
      if(checked?, do: [:out, :ok], else: :out)
    )
  end

  defp widened_pdl(config) do
    for op <- config.widened_ops, prefix <- ["", "try_"] do
      name = "#{prefix}int_#{op}"
      outs = if prefix == "", do: :out, else: [:out, :ok]
      binary_spec(config, name, prefix != "", [c(config, outs, "#{name}_i64", [:x, :y], types(outs))])
    end
  end

  defp scalar_arith_pdl(config) do
    for {op, arith_op} <- config.scalar_arith do
      symbol = "scalar_#{op}"
      binary_spec(config, symbol, false, [v(:sum, arith_op, [:x, :y], "i64"), c(config, :out, "box_int", [:sum])])
    end
  end

  defp cmp_pdl(config) do
    for {name, code} <- config.cmp_preds, prefix <- ~w(int scalar try_int) do
      symbol = "#{prefix}_cmp_#{name}"
      checked? = prefix == "try_int"

      rewrite =
        [v(:cmp, "cmpi", [:x, :y], "i1", predicate: "#{code} : i64"), c(config, :out, "box_bool_i1", [:cmp])] ++
          if(checked?, do: [constant(:ok, "true", "i1")], else: [])

      binary_spec(config, symbol, checked?, rewrite)
    end
  end
end
