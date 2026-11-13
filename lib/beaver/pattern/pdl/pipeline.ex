defmodule Beaver.Pattern.PDL.Pipeline do
  @moduledoc """
  Staged call-carrier lowering. The runtime provider supplies function names, layers, symbols and signatures;
  the template provider supplies declarations and inline/runtime imports. Policy stays in explicit options.
  Declaration checks precede rewriting; specialization revisits newly introduced calls before import.
  """
  alias Beaver.{Composer, MLIR}
  alias Beaver.MLIR.{Rewrite, SymbolTable}
  alias Beaver.Pattern.PDL
  @dissolve ~w(inline sccp canonicalize cse symbol-dce)
  @cleanup ~w(sccp canonicalize cse symbol-dce)

  @doc "Defines a lowering facade over a rule supplier, runtime provider, template provider and policy."
  defmacro __using__(opts) do
    runtime = Keyword.fetch!(opts, :runtime)
    templates = Keyword.fetch!(opts, :templates)
    policy = Keyword.fetch!(opts, :policy)
    rules = Keyword.fetch!(opts, :rules)

    quote do
      def core_pdl_source, do: Beaver.Pattern.PDL.source(unquote(rules).())

      def c_runtime_pdl_source(ns, functions \\ unquote(runtime).functions()),
        do: unquote(__MODULE__).bindings_source(unquote(runtime), ns, functions, unquote(policy))

      def declaration(opts \\ []),
        do: unquote(__MODULE__).declaration(unquote(rules).(), unquote(runtime), opts, unquote(policy))

      def lower(module, opts \\ []) when is_list(opts) do
        {:ok, lower!(module, opts)}
      rescue
        e -> {:error, e}
      end

      def lower!(module, opts \\ []),
        do:
          unquote(__MODULE__).lower!(
            module,
            unquote(rules).(),
            unquote(templates),
            unquote(runtime),
            opts,
            unquote(policy)
          )

      def specialize!(module), do: unquote(__MODULE__).specialize!(unquote(rules).(), unquote(templates), module)
    end
  end

  def declaration(rules, runtime, opts, policy) do
    layers = Keyword.get(opts, :layers, policy[:layers]) |> List.wrap()

    if layers == [] or layers -- policy[:layers] != [] or Enum.uniq(layers) != layers,
      do: raise(ArgumentError, "invalid layers: #{inspect(opts[:layers])}")

    ns = opts[:runtime_namespace]
    if ns != nil, do: runtime.validate_namespace!(ns)
    %{layers: layers, runtime_namespace: ns, pdl_sha256: rules.digest}
  end

  def bindings_source(runtime, ns, functions, policy) do
    ns = runtime.validate_namespace!(ns)

    for name <- Enum.uniq(functions) do
      PDL.Source.retarget(
        policy[:rule_prefix] <> name,
        runtime.raw_symbol(name),
        runtime.symbol(name, ns),
        policy[:results],
        2
      )
    end
    |> PDL.new()
    |> PDL.source()
  end

  def specialize!(rules, templates, module),
    do: elem(specialize(rules, templates, module, SymbolTable.referenced_symbols(module)), 0)

  defp specialize(rules, templates, module, refs) do
    if PDL.applicable?(rules, refs) do
      PDL.apply!(rules, module)
      templates.ensure_declarations!(module)
      {module, SymbolTable.referenced_symbols(module), true}
    else
      {module, refs, false}
    end
  end

  def lower!(module, rules, templates, runtime, opts, policy) do
    %{layers: layers, runtime_namespace: ns} = declaration(rules, runtime, opts, policy)
    refs = SymbolTable.referenced_symbols(module)
    {selected, declarations} = selection!(module, refs, runtime, ns, layers, opts, policy)

    active? = active?(refs, layers, policy)
    {module, refs, changed?} = if active?, do: specialize(rules, templates, module, refs), else: {module, refs, false}

    imports = imports(refs, layers, selected, runtime, policy)
    if imports != [], do: templates.import_inline!(module, symbols: imports)

    if selected != [] do
      SymbolTable.ensure_declarations!(module, declarations)
      Rewrite.apply_patterns!(module, [{:pdl, bindings_source(runtime, ns, selected, policy)}])
    end

    passes = passes(Keyword.get(opts, :dissolve, :auto), imports != [], changed? or selected != [])
    if passes != [], do: Enum.reduce(passes, module, &Composer.append(&2, &1)) |> Composer.run!()
    if opts[:import_runtime], do: templates.import_runtime!(module)
    MLIR.verify!(module)
  end

  defp selection!(module, refs, runtime, ns, layers, opts, policy) do
    bindings = Map.new(for f <- runtime.functions(), s <- [runtime.raw_symbol(f), runtime.value_symbol(f)], do: {s, f})
    all = refs |> Enum.map(&bindings[&1]) |> Enum.reject(&is_nil/1) |> Enum.uniq()
    selected = Enum.filter(all, &(runtime.layer(&1) in layers))

    if selected != [] and is_nil(ns),
      do: raise(ArgumentError, "missing :runtime_namespace for #{policy[:label]} calls: #{inspect(selected)}")

    if selected == [] and all != [] and opts[:require_full],
      do: raise(ArgumentError, "unselected #{policy[:label]} layer calls remain: #{inspect(all)}")

    declarations = for f <- selected, do: {runtime.symbol(f, ns), runtime.function_type(f)}
    SymbolTable.check_declarations!(module, declarations)

    {selected, declarations}
  end

  defp active?(refs, layers, policy),
    do:
      Enum.any?(layers, &(&1 in policy[:specialize_layers])) and
        Enum.any?(refs, &String.starts_with?(&1, policy[:prefix]))

  defp imports(refs, layers, selected, runtime, policy) do
    imports =
      for ref <- refs,
          policy[:import_layer] in layers,
          String.starts_with?(ref, policy[:prefix]),
          ref not in policy[:reserved],
          do: ref

    imports ++ for(f <- selected, symbol = runtime.value_symbol(f), symbol in refs, do: symbol)
  end

  defp passes(true, _, _), do: @dissolve
  defp passes(false, _, _), do: []
  defp passes(:auto, true, _), do: @dissolve
  defp passes(:auto, false, true), do: @cleanup
  defp passes(:auto, false, false), do: []
end
