alias Beaver.MLIR.{Attribute, CAPI, StringRef}
alias Beaver.{Deferred, Native}

defmodule Beaver.Slang do
  use Beaver
  alias Beaver.MLIR.Dialect.IRDL
  @variadic_tags [:variadic, :optional, :single]
  @callback __slang_dialect__(ctx :: Beaver.MLIR.Context.t()) :: Beaver.MLIR.Module.t()
  @callback __slang_traits__() :: [{String.t(), [Beaver.MLIR.Trait.declaration()]}]
  @callback __slang_interfaces__() :: [{String.t(), keyword()}]
  @callback __slang_dialect_name__() :: String.t()
  @callback __slang_schema_digest__() :: String.t()
  @moduledoc """
  Defines extensible MLIR dialects in Elixir and compiles their schemas to
  [IRDL](https://mlir.llvm.org/docs/Dialects/IRDL/).

  Slang supports named type and attribute parameters, named operation operands,
  results, attributes and regions, reusable constraints, `any_of`, `all_of` and
  `base` constraints, and optional or variadic operands and results. Built-in
  and callback-backed custom operation traits can be attached when
  the dialect is loaded.

  Schema construction and runtime interface attachment are separate:
  `__slang_dialect__/1` builds the inspectable IRDL module, while `load/2`
  verifies and registers that module before attaching traits. See the
  [Slang guide](slang.html) for a complete dialect and the supported syntax.
  """

  @doc "Registers the dialect declarations and imports the Slang macros."
  defmacro __using__(opts) do
    name = opts |> Keyword.fetch!(:name)

    quote do
      @behaviour Beaver.Slang
      @before_compile Beaver.Slang
      import Beaver.Slang, only: :macros
      @__slang_dialect_name__ unquote(name)
      Module.register_attribute(__MODULE__, :__slang__operation__, accumulate: true)
      Module.register_attribute(__MODULE__, :__slang__creator__, accumulate: true)
      Module.register_attribute(__MODULE__, :__slang__type__, accumulate: true)
      Module.register_attribute(__MODULE__, :__slang__trait__, accumulate: true)
      Module.register_attribute(__MODULE__, :__slang__interface__, accumulate: true)
      Module.register_attribute(__MODULE__, :__slang__op_meta__, accumulate: true)
    end
  end

  @doc false
  defmacro __before_compile__(_env) do
    introspection_ast = slang_introspection_ast()

    quote do
      @__slang_ordered_ops__ Beaver.Slang.__reverse_list__(@__slang__operation__)
      @__slang_ordered_meta__ Beaver.Slang.__reverse_list__(@__slang__op_meta__)

      @doc false
      def __slang_dialect__(ctx),
        do: Beaver.Slang.create_dialect(@__slang_dialect_name__, Enum.reverse(@__slang__creator__), ctx: ctx)

      @doc false
      def __slang_traits__, do: Enum.reverse(@__slang__trait__)

      @doc false
      def __slang_interfaces__, do: Enum.reverse(@__slang__interface__)

      @doc false
      def __slang_dialect_name__, do: @__slang_dialect_name__

      unquote(introspection_ast)

      use Beaver.MLIR.Dialect,
        dialect: @__slang_dialect_name__,
        ops: Beaver.Slang.__or_empty__(@__slang__operation__)
    end
  end

  defp slang_introspection_ast do
    quote do
      @doc "Returns a deterministic SHA-256 digest of the canonicalized IRDL dialect schema."
      def __slang_schema_digest__, do: Beaver.Slang.schema_digest(__MODULE__)

      @doc "Returns full operation names (`\"dialect.op\"`) declared in this dialect."
      def __slang_operations__, do: Beaver.Slang.__ops_for_dialect__(@__slang_dialect_name__, @__slang_ordered_ops__)

      @doc "Returns full operation names (`\"dialect.op\"`) belonging to `layer`."
      def __slang_operations__(layer),
        do: Beaver.Slang.__ops_for_layer__(@__slang_dialect_name__, @__slang_ordered_meta__, layer)

      @doc "Returns unique non-nil layers declared on operations in this dialect."
      def __slang_layers__, do: Beaver.Slang.__layers_for_meta__(@__slang_ordered_meta__)

      @doc "Returns the declared `:layer` for `op_name`, or `nil`."
      def __slang_layer__(op_name),
        do: Beaver.Slang.__layer_for_op__(@__slang_dialect_name__, @__slang_ordered_meta__, op_name)

      @doc "Returns the declared `:metadata` map for `op_name`."
      def __slang_metadata__(op_name),
        do: Beaver.Slang.__metadata_for_op__(@__slang_dialect_name__, @__slang_ordered_meta__, op_name)
    end
  end

  defp creator_env do
    quote do
      [ip: Beaver.Env.block(), ctx: Beaver.Env.context(), loc: Kernel.var!(slang_internal_source_loc)]
    end
  end

  @doc false
  defp transform_defop_pins({:^, _line1, [{name, _line2, ctx}]}) when is_atom(name) and is_atom(ctx) do
    alias_name = get_alias_name(name)

    quote do
      unquote(alias_name)(unquote(creator_env()))
    end
  end

  defp transform_defop_pins({:=, _line0, [_var, {:=, _line1, _right2}]} = ast), do: ast
  # only leaf assignment should be transformed
  defp transform_defop_pins({:=, _line0, [var, right]}) do
    quote do
      unquote(var) =
        Beaver.Slang.create_constraint(
          unquote(right),
          unquote(creator_env())
        )
    end
  end

  defp transform_defop_pins(ast), do: ast

  @doc false
  def create_constraint({variadic_tag, v}, opts) when variadic_tag in @variadic_tags,
    do: {variadic_tag, create_constraint(v, opts)}

  def create_constraint(%MLIR.Value{} = v, _opts), do: v

  def create_constraint(t, opts) do
    use Beaver

    mlir ctx: opts[:ctx], ip: Deferred.fetch_insertion_point(opts) do
      loc = constraint_location(opts, Beaver.Env.context())
      Beaver.MLIR.Dialect.IRDL.is([loc, expected: t]) >>> ~t{!irdl.attribute}
    end
  end

  defp emit_declaration(op, values, names, opts, extra) do
    n = Enum.count(values, &match?(%MLIR.Value{}, &1))

    if length(names) != n,
      do: raise(ArgumentError, "#{op} declares #{length(names)} names for #{n} constraints: #{inspect(names)}")

    key = if op == :attributes, do: :attributeValueNames, else: :names

    attributes =
      if op in [:operands, :results, :parameters, :regions, :attributes],
        do: [{key, Attribute.array(Enum.map(names, &Attribute.string/1))}],
        else: []

    MLIR.Operation.create(%Beaver.SSA{
      op: "irdl.#{op}",
      arguments: values ++ attributes ++ extra,
      ctx: opts[:ctx],
      ip: opts[:ip],
      loc: opts[:loc]
    })
  end

  defp get_variadicity(values, opts) do
    if opts[:need_variadicity] do
      tags =
        Enum.map(List.wrap(values), fn
          {tag, _} when tag in @variadic_tags -> %{single: 0, optional: 1, variadic: 2}[tag]
          _ -> 0
        end)

      data = Native.array(tags, Native.I32)
      [variadicity: Deferred.defer(&CAPI.beaverIRDLVariadicityArrayAttrGet(&1, length(tags), data))]
    else
      []
    end
  end

  defp strip_variadicity(values) do
    values
    |> List.wrap()
    |> Enum.map(fn
      {variadic_tag, v} when variadic_tag in @variadic_tags -> v
      v -> v
    end)
  end

  @doc false
  # Build constraint groups in an owned region, then attach the enclosing IRDL declaration.
  def run_creator(name, op, args_op, constrain_f, opts) do
    Deferred.from_opts(opts, fn ctx ->
      location = source_location(opts, ctx)
      region = MLIR.Region.create()
      block = MLIR.Block.create()
      MLIR.Region.append(region, block)
      local = [ctx: ctx, ip: block, loc: location]

      # The region is still caller-owned until the enclosing declaration is created.
      try do
        {args, ret, attributes, regions} = constrain_f.(block, ctx)

        groups = [
          {args_op, args, opts[:argument_names], opts},
          {opts[:return_op], ret, opts[:result_names], opts},
          {:attributes, attributes, opts[:attribute_names], []},
          {:regions, regions, opts[:region_names], []}
        ]

        for {group, descriptors, names, variadic_opts} <- groups,
            values = strip_variadicity(descriptors),
            values != [] do
          if is_nil(group), do: raise(CaseClauseError, term: {group, descriptors, values})
          values = if group == opts[:return_op], do: Enum.map(values, &create_constraint(&1, local)), else: values
          emit_declaration(group, values, names, local, get_variadicity(descriptors, variadic_opts))
        end
      catch
        kind, reason ->
          MLIR.Region.destroy(region)
          :erlang.raise(kind, reason, __STACKTRACE__)
      end

      emit_declaration(op, [region], [], Keyword.put(local, :ip, Deferred.fetch_insertion_point(opts)),
        sym_name: Attribute.string(name, ctx: ctx)
      )
    end)
  end

  defp source_location(opts, ctx) do
    case opts[:source] do
      %{file: file, line: line} -> Deferred.resolve(MLIR.Location.file(name: file, line: line), ctx)
      _ -> Deferred.resolve(MLIR.Location.unknown(), ctx)
    end
  end

  @doc false
  def constraint_location(opts, ctx), do: opts[:loc] || source_location(opts, ctx)

  defp get_slang_arg_ast(i), do: {"slang_internal_arg#{i}" |> String.to_atom(), [], nil}

  defp transform_constraint(ast, i) do
    case ast do
      {name, _line0, ctx} when is_atom(name) and is_atom(ctx) ->
        nil

      {:=, _line0, [_var, _right]} ->
        ast

      {variadic_tag, {name, _line0, ctx}} when variadic_tag in @variadic_tags and is_atom(name) and is_atom(ctx) ->
        ast

      _ ->
        quote do
          unquote(get_slang_arg_ast(i)) =
            Beaver.Slang.create_constraint(
              unquote(ast),
              unquote(creator_env())
            )
        end
    end
  end

  defp transform_variable(ast, i) do
    case ast do
      {name, _line0, ctx} when is_atom(name) and is_atom(ctx) -> ast
      {:=, _line0, [var, _right]} -> var
      {variadic_tag, {name, _line0, ctx}} when variadic_tag in @variadic_tags and is_atom(name) and is_atom(ctx) -> ast
      _ -> get_slang_arg_ast(i)
    end
  end

  defp get_args_as_vars(args) do
    for {v, i} <- Enum.with_index(args), do: transform_variable(v, i)
  end

  defp declaration_name({:=, _, [var, _]}, index, prefix), do: declaration_name(var, index, prefix)

  defp declaration_name({tag, value}, index, prefix) when tag in @variadic_tags,
    do: declaration_name(value, index, prefix)

  defp declaration_name({name, _, ctx}, _index, _prefix) when is_atom(name) and is_atom(ctx),
    do: Atom.to_string(name)

  defp declaration_name(_ast, index, prefix), do: "#{prefix}_#{index + 1}"

  defp normalize_names!(asts, nil, prefix) do
    asts
    |> Enum.with_index()
    |> Enum.map(fn {ast, index} -> declaration_name(ast, index, prefix) end)
    |> uniquify_names()
  end

  defp normalize_names!(asts, names, _prefix) when is_list(names) do
    names = Enum.map(names, &normalize_name!/1)

    if length(asts) != length(names),
      do: raise(ArgumentError, "expected #{length(asts)} names, got #{length(names)}: #{inspect(names)}")

    if length(Enum.uniq(names)) != length(names),
      do: raise(ArgumentError, "Slang declaration names must be unique: #{inspect(names)}")

    names
  end

  defp normalize_names!(_asts, names, _prefix),
    do: raise(ArgumentError, "expected a list of Slang declaration names, got: #{inspect(names)}")

  defp normalize_name!(name) when is_atom(name), do: Atom.to_string(name)
  defp normalize_name!(name) when is_binary(name) and name != "", do: name

  defp normalize_name!(name),
    do: raise(ArgumentError, "expected a non-empty atom or string name, got: #{inspect(name)}")

  defp uniquify_names(names) do
    {names, _used} =
      Enum.map_reduce(names, MapSet.new(), fn name, used ->
        unique_name = available_name(name, used, 1)
        {unique_name, MapSet.put(used, unique_name)}
      end)

    names
  end

  defp available_name(name, used, suffix) do
    candidate = if suffix == 1, do: name, else: "#{name}_#{suffix}"

    if MapSet.member?(used, candidate),
      do: available_name(name, used, suffix + 1),
      else: candidate
  end

  defp uniquify_declaration_names(arg_names, result_names, attribute_names, region_names) do
    names = uniquify_names(arg_names ++ result_names ++ attribute_names ++ region_names)

    {arg_names, names} = Enum.split(names, length(arg_names))
    {result_names, names} = Enum.split(names, length(result_names))
    {attribute_names, region_names} = Enum.split(names, length(attribute_names))

    {arg_names, result_names, attribute_names, region_names}
  end

  defp split_named_values(nil, _prefix, _explicit_names), do: {[], []}

  defp split_named_values(values, prefix, explicit_names) when is_list(values) do
    if named_value_keyword?(values) do
      if explicit_names,
        do: raise(ArgumentError, "cannot combine named #{prefix} entries with an explicit #{prefix}_names option")

      {Enum.map(values, &elem(&1, 1)), Enum.map(values, &(&1 |> elem(0) |> normalize_name!()))}
    else
      {values, normalize_names!(values, explicit_names, prefix)}
    end
  end

  defp split_named_values(value, prefix, explicit_names) do
    values = [value]
    {values, normalize_names!(values, explicit_names, prefix)}
  end

  defp named_value_keyword?(values) do
    Keyword.keyword?(values) and Enum.all?(values, fn {name, _value} -> name not in @variadic_tags end)
  end

  defp normalize_results_ast({:__block__, meta, expressions}, explicit_names) do
    {result_expression, result_names} = normalize_results_ast(List.last(expressions), explicit_names)
    expressions = List.replace_at(expressions, -1, result_expression)
    {{:__block__, meta, expressions}, result_names}
  end

  defp normalize_results_ast(results, explicit_names) do
    {result_values, result_names} = split_named_values(results, "result", explicit_names)
    {quote(do: [unquote_splicing(result_values)]), result_names}
  end

  defp split_attributes(nil), do: {[], []}

  defp split_attributes(attributes) when is_list(attributes) do
    if Keyword.keyword?(attributes),
      do: {Enum.map(attributes, &elem(&1, 1)), Enum.map(attributes, &(&1 |> elem(0) |> normalize_name!()))},
      else: raise(ArgumentError, "operation attributes must be a keyword list of name: constraint entries")
  end

  defp split_attributes(attributes),
    do: raise(ArgumentError, "operation attributes must be a keyword list, got: #{Macro.to_string(attributes)}")

  defp normalize_region_descriptor(:any), do: %{args: nil, size: nil}

  defp normalize_region_descriptor({:sized, size}) when is_integer(size) and size > 0, do: %{args: nil, size: size}

  defp normalize_region_descriptor({:region, opts}) when is_list(opts) do
    opts = Keyword.validate!(opts, [:args, :size])

    args =
      case Keyword.get(opts, :args) do
        nil -> nil
        args -> List.wrap(args)
      end

    size =
      case Keyword.get(opts, :size) do
        nil ->
          nil

        size when is_integer(size) and size > 0 ->
          size

        size ->
          raise ArgumentError,
                "expected region size to be a positive integer, got: #{inspect(size)}"
      end

    %{args: args, size: size}
  end

  defp build_region_arguments(%{args: args, size: size}, block, ctx, loc) do
    constrained_args? = not is_nil(args)

    args =
      for arg <- args || [] do
        create_constraint(arg, ip: block, ctx: ctx, loc: loc)
      end

    attrs =
      if is_nil(size),
        do: [],
        else: [numberOfBlocks: Deferred.resolve(Attribute.integer(MLIR.Type.i32(), size), ctx)]

    attrs =
      if constrained_args?, do: attrs ++ [constrainedArguments: Deferred.resolve(Attribute.unit(), ctx)], else: attrs

    args ++ attrs
  end

  defp do_gen_region(region, block, ctx, loc) do
    use Beaver

    region_args = region |> normalize_region_descriptor() |> build_region_arguments(block, ctx, loc)

    %Beaver.SSA{
      arguments: region_args,
      results: [Deferred.resolve(~t{!irdl.region}, ctx)],
      ip: block,
      ctx: ctx,
      loc: loc,
      evaluator: &MLIR.Operation.eval_ssa/1
    }
    |> IRDL.region()
  end

  @doc false
  def gen_region(regions, block, ctx, loc),
    do: regions |> List.wrap() |> Enum.map(&do_gen_region(&1, block, ctx, loc))

  @doc false
  defp gen_creator(op, args_op, call, do_block, opts) do
    {name, args} = call |> Macro.decompose_call()
    name = Atom.to_string(name)
    creator = String.to_atom("create_" <> name)
    attr_name = String.to_atom("__slang__" <> "#{op}" <> "__")
    args_var_ast = get_args_as_vars(args)
    argument_names = normalize_names!(args, opts[:argument_names], Atom.to_string(args_op))

    {result_ast, result_names} = normalize_results_ast(do_block, opts[:result_names])
    result_ast = Macro.postwalk(result_ast, &transform_defop_pins/1)
    {attribute_values, attribute_names} = split_attributes(opts[:attributes])
    {region_values, region_names} = split_named_values(opts[:regions], "region", nil)

    {argument_names, result_names, attribute_names, region_names} =
      uniquify_declaration_names(argument_names, result_names, attribute_names, region_names)

    input_constrains =
      args
      |> Macro.postwalk(&transform_defop_pins/1)
      |> Enum.with_index()
      |> Enum.map(fn {ast, i} -> transform_constraint(ast, i) end)

    {attribute_constraints, attribute_vars} =
      attribute_values
      |> Macro.postwalk(&transform_defop_pins/1)
      |> Enum.with_index(length(args))
      |> Enum.map_reduce([], fn {ast, i}, vars ->
        {transform_constraint(ast, i), vars ++ [transform_variable(ast, i)]}
      end)

    creator_opts =
      opts
      |> Keyword.drop(~w(argument_names result_names attributes regions traits interfaces layer metadata)a)
      |> Keyword.merge(
        argument_names: argument_names,
        result_names: result_names,
        attribute_names: attribute_names,
        region_names: region_names
      )

    traits = Beaver.MLIR.Trait.normalize!(opts[:traits])
    interfaces = normalize_interfaces!(opts[:interfaces])

    quote do
      Module.put_attribute(__MODULE__, unquote(attr_name), unquote(name))
      @__slang__creator__ {unquote(op), __MODULE__, unquote(creator)}
      unquote_splicing(
        if op == :operation do
          [
            quote do
              @__slang__op_meta__ {unquote(name), unquote(opts[:layer]),
                                   Beaver.Slang.__normalize_op_metadata__!(unquote(opts[:metadata]))}
            end
          ]
        else
          []
        end
      )

      unquote_splicing(
        for {key, values} <- [__slang__trait__: traits, __slang__interface__: interfaces], values != [] do
          quote do
            Module.put_attribute(__MODULE__, unquote(key), {unquote(name), unquote(values)})
          end
        end
      )

      def unquote(creator)(opts) do
        Beaver.Slang.run_creator(
          unquote(name),
          unquote(op),
          unquote(args_op),
          fn block, ctx ->
            use Beaver

            mlir ip: block, ctx: ctx do
              Kernel.var!(slang_internal_source_loc) =
                Beaver.Slang.constraint_location(unquote(Macro.escape(creator_opts)), ctx)

              unquote_splicing(input_constrains)
              unquote_splicing(attribute_constraints)

              regions =
                Beaver.Slang.gen_region(unquote(region_values), block, ctx, Kernel.var!(slang_internal_source_loc))

              {[unquote_splicing(args_var_ast)], unquote(result_ast), [unquote_splicing(attribute_vars)], regions}
            end
          end,
          opts ++ unquote(Macro.escape(creator_opts))
        )
      end
    end
  end

  @doc false
  def create_constrained_element(element, dialect, name, params, opts \\ []) do
    Deferred.from_opts(opts, fn ctx ->
      params =
        params
        |> Enum.map(&Deferred.resolve(&1, ctx))
        |> Enum.map(fn
          %MLIR.Type{} = t -> Attribute.type(t)
          %Attribute{} = a -> a
        end)
        |> Attribute.array(ctx: ctx)

      apply(
        CAPI,
        case element do
          :attribute -> :beaverIRDLGetDefinedAttr
          :type -> :beaverIRDLGetDefinedType
        end,
        [StringRef.create(dialect), StringRef.create(name), params]
      )
    end)
  end

  defp gen_create_element_creator(element, call, opts, caller) do
    {name, args} =
      call |> Macro.decompose_call()

    opts = Keyword.validate!(opts, [:parameter_names])
    source = %{file: caller.file, line: caller.line}

    [
      gen_creator(element, :parameters, call, nil,
        need_variadicity: false,
        argument_names: opts[:parameter_names],
        source: source
      ),
      quote do
        def unquote(name)(unquote_splicing(get_args_as_vars(args)), opts \\ []) do
          {:parametric, Attribute.symbol_ref(@__slang_dialect_name__, [to_string(unquote(name))]),
           [unquote_splicing(get_args_as_vars(args))],
           Beaver.Slang.create_constrained_element(
             unquote(element),
             @__slang_dialect_name__,
             "#{unquote(name)}",
             [unquote_splicing(get_args_as_vars(args))],
             opts
           )}
        end
      end
    ]
  end

  @doc "Defines a type in the dialect."
  defmacro deftype(call, opts \\ []),
    do: gen_create_element_creator(:type, call, opts, __CALLER__)

  @doc "Defines an attribute in the dialect."
  defmacro defattr(call, opts \\ []),
    do: gen_create_element_creator(:attribute, call, opts, __CALLER__)

  @doc "Defines an operation and generates its creator function."
  defmacro defop(call, block \\ nil) do
    block =
      Keyword.validate!(
        block || [],
        ~w(do results result_names operand_names attributes regions traits interfaces layer metadata)a
      )

    if Keyword.has_key?(block, :do) and Keyword.has_key?(block, :results),
      do: raise(ArgumentError, "defop accepts either a do block or :results, not both")

    results = Keyword.get(block, :results, block[:do])

    gen_creator(:operation, :operands, call, results,
      return_op: :results,
      need_variadicity: true,
      argument_names: block[:operand_names],
      result_names: block[:result_names],
      attributes: block[:attributes],
      regions: block[:regions],
      traits: block[:traits],
      interfaces: block[:interfaces],
      layer: block[:layer],
      metadata: block[:metadata],
      source: %{file: __CALLER__.file, line: __CALLER__.line}
    )
  end

  @doc """
  Defines multiple operations declaratively from a table of specs without manual AST construction.

  Each entry in `specs` may be `{name, operands, results}`, `{name, operands, results, attributes}`,
  `{name, operands, results, attributes, op_opts}`, or a keyword/map spec. Constraint atoms like `:tv`
  expand to pinned named constraints `^tv`, and `{:variadic, c}` / `{:optional, c}` / `{:base, b}` /
  `:any` expand to the corresponding Slang constraint forms.
  """
  defmacro defops(specs, shared_opts \\ []) do
    expanded_specs = Macro.expand(specs, __CALLER__)
    expanded_shared = Macro.expand(shared_opts, __CALLER__)

    unless is_list(expanded_specs),
      do: raise(ArgumentError, "defops expects a list of operation specifications, got: #{Macro.to_string(specs)}")

    decls = Enum.map(expanded_specs, &expand_table_op!(&1, expanded_shared))

    quote do
      (unquote_splicing(decls))
    end
  end

  defp get_alias_name(def_name), do: String.to_atom("alias_" <> Atom.to_string(def_name))

  defp gen_constraint(call, block) do
    {name, args} = call |> Macro.decompose_call()

    if args != [], do: raise(ArgumentError, "named Slang constraints do not accept arguments")

    alias_name = get_alias_name(name)
    block = block |> Macro.postwalk(&transform_defop_pins/1)

    quote do
      @slang_alias unquote(alias_name)
      def unquote(alias_name)(opts) do
        use Beaver

        mlir ctx: opts[:ctx], ip: Deferred.fetch_insertion_point(opts) do
          Kernel.var!(slang_internal_source_loc) =
            Beaver.Slang.constraint_location(opts, Beaver.Env.context())

          unquote(block[:do])
          |> Beaver.Slang.create_parametric(unquote(creator_env()))
        end
      end
    end
  end

  @doc """
  Defines a reusable named constraint.

  Reference it from another declaration with `^name`.
  """
  defmacro defconstraint(call, block), do: gen_constraint(call, block)

  @doc """
  Defines a reusable named constraint.

  `defalias` is retained as a concise spelling of `defconstraint`.
  """
  defmacro defalias(call, block), do: gen_constraint(call, block)

  @doc "Generates an `irdl.any_of` constraint."
  defmacro any_of(types), do: combine_constraints(:any_of, types)

  @doc "Generates an `irdl.all_of` constraint."
  defmacro all_of(types), do: combine_constraints(:all_of, types)

  defp combine_constraints(operation, types) do
    quote do
      use Beaver

      mlir do
        types =
          for type <- unquote(types) do
            Beaver.Slang.create_constraint(
              type,
              unquote(creator_env())
            )
          end

        Beaver.MLIR.Dialect.IRDL.unquote(operation)([Kernel.var!(slang_internal_source_loc) | types]) >>>
          ~t{!irdl.attribute}
      end
    end
  end

  @doc false
  def create_base({:parametric, symbol, _values, _element}, opts),
    do: create_base({:base_ref, symbol}, opts)

  def create_base(name, opts) when is_binary(name), do: create_base({:base_name, Attribute.string(name)}, opts)

  def create_base({key, value}, opts) when key in [:base_ref, :base_name] do
    Deferred.from_opts(opts, fn ctx ->
      mlir ctx: ctx, ip: Deferred.fetch_insertion_point(opts) do
        loc = constraint_location(opts, ctx)

        IRDL.base([loc, {key, Deferred.resolve(value, ctx)}]) >>>
          ~t{!irdl.attribute}
      end
    end)
  end

  @doc """
  Generates an `irdl.base` constraint from a base name or a Slang type/attribute.

  Base names use upstream IRDL spelling such as `"!builtin.integer"` or
  `"#builtin.string"`.
  """
  defmacro base(base) do
    quote do
      Beaver.Slang.create_base(
        unquote(base),
        unquote(creator_env())
      )
    end
  end

  @doc "Marks an operand or result constraint as variadic."
  defmacro variadic(constraint), do: quote(do: {:variadic, unquote(constraint)})

  @doc "Marks an operand or result constraint as optional."
  defmacro optional(constraint), do: quote(do: {:optional, unquote(constraint)})

  @doc "Marks an operand or result constraint as a required singleton."
  defmacro single(constraint), do: quote(do: {:single, unquote(constraint)})

  @doc "Generates an `irdl.is` constraint, typically for a type."
  defmacro is(type) do
    quote do
      use Beaver

      mlir do
        Beaver.MLIR.Dialect.IRDL.is([Kernel.var!(slang_internal_source_loc), expected: unquote(type)]) >>>
          ~t{!irdl.attribute}
      end
    end
  end

  @doc "Generates an `irdl.any` constraint."
  defmacro any() do
    quote do
      use Beaver

      mlir do
        Beaver.MLIR.Dialect.IRDL.any(Kernel.var!(slang_internal_source_loc)) >>> ~t{!irdl.attribute}
      end
    end
  end

  @doc false
  def create_parametric({:parametric, symbol, values, _}, opts) do
    base_type = symbol

    Deferred.from_opts(
      opts,
      fn ctx ->
        mlir ctx: ctx, ip: Deferred.fetch_insertion_point(opts) do
          loc = constraint_location(opts, ctx)
          IRDL.parametric([loc | values], base_type: base_type) >>> ~t{!irdl.attribute}
        end
      end
    )
  end

  @doc false
  def create_parametric(v, _opts), do: v

  @doc false
  def create_dialect(name, creators, opts) do
    Deferred.from_opts(
      opts,
      fn ctx ->
        mlir ctx: ctx do
          module do
            IRDL.dialect sym_name: ~a/#{name}/s do
              region do
                block _dialect() do
                  for {_type, m, f} <- creators do
                    opts = [ctx: Beaver.Env.context(), ip: Beaver.Env.block()]
                    apply(m, f, [opts])
                  end
                end
              end
            end >>> []
          end
        end
      end
    )
  end

  @external_interfaces ~w(memory_effects conditionally_speculatable transform_op pattern_descriptor)a

  defp normalize_interfaces!(nil), do: []

  defp normalize_interfaces!(interfaces) when is_list(interfaces) do
    unless Keyword.keyword?(interfaces),
      do: raise(ArgumentError, ":interfaces must be a keyword list")

    unsupported = Keyword.keys(interfaces) -- @external_interfaces

    if unsupported != [], do: raise(ArgumentError, "unsupported Slang interfaces: #{inspect(unsupported)}")

    interfaces
  end

  defp normalize_interfaces!(other),
    do: raise(ArgumentError, ":interfaces must be a keyword list, got: #{inspect(other)}")

  @doc false
  def __normalize_op_metadata__!(nil), do: %{}
  def __normalize_op_metadata__!(%{} = map), do: map
  def __normalize_op_metadata__!(kw) when is_list(kw), do: Map.new(kw)

  def __normalize_op_metadata__!(other),
    do: raise(ArgumentError, "operation :metadata must be a map or keyword list, got: #{inspect(other)}")

  @doc false
  def __or_empty__(nil), do: []
  def __or_empty__(list) when is_list(list), do: list

  @doc false
  def __reverse_list__(nil), do: []
  def __reverse_list__(list) when is_list(list), do: Enum.reverse(list)

  @doc false
  def __ops_for_dialect__(dialect_name, ordered_ops),
    do: Enum.map(ordered_ops, &(dialect_name <> "." <> &1))

  @doc false
  def __ops_for_layer__(dialect_name, ordered_meta, layer) do
    for {op, ^layer, _meta} <- ordered_meta, do: dialect_name <> "." <> op
  end

  @doc false
  def __layers_for_meta__(ordered_meta) do
    ordered_meta
    |> Enum.map(&elem(&1, 1))
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
  end

  @doc false
  def __layer_for_op__(dialect_name, ordered_meta, op_name) do
    short = short_op_name(dialect_name, op_name)

    case List.keyfind(ordered_meta, short, 0) do
      {^short, layer, _meta} -> layer
      nil -> nil
    end
  end

  @doc false
  def __metadata_for_op__(dialect_name, ordered_meta, op_name) do
    short = short_op_name(dialect_name, op_name)

    case List.keyfind(ordered_meta, short, 0) do
      {^short, _layer, meta} -> meta
      nil -> %{}
    end
  end

  @doc false
  def short_op_name(dialect_name, op_name) do
    str = to_string(op_name)
    prefix = dialect_name <> "."
    String.replace_prefix(str, prefix, "")
  end

  @doc """
  Computes (and caches per compiled module MD5 via `:persistent_term`) a deterministic
  `"sha256:..."` digest of `mod`'s canonicalized IRDL schema, traits, and interface keys.
  """
  @spec schema_digest(module()) :: String.t()
  def schema_digest(mod) when is_atom(mod) do
    cache_key = {__MODULE__, :schema_digest, mod, mod.__info__(:md5)}

    case :persistent_term.get(cache_key, nil) do
      digest when is_binary(digest) ->
        digest

      nil ->
        digest = compute_schema_digest(mod)
        :persistent_term.put(cache_key, digest)
        digest
    end
  end

  defp compute_schema_digest(mod) do
    ir_text =
      MLIR.Context.with_context(fn ctx ->
        mod
        |> apply(:__slang_dialect__, [ctx])
        |> Beaver.MLIR.Transform.canonicalize()
        |> Beaver.Composer.run!()
        |> MLIR.verify!()
        |> MLIR.Module.with_module(fn irdl_mod ->
          MLIR.to_string(irdl_mod, generic: true)
        end)
      end)

    trait_summary =
      Enum.map(mod.__slang_traits__(), fn {op, decls} ->
        {op,
         Enum.map(decls, fn
           {id, kw} when is_list(kw) -> {id, Keyword.keys(kw)}
           builtin -> builtin
         end)}
      end)

    iface_summary =
      Enum.map(mod.__slang_interfaces__(), fn {op, kw} -> {op, Keyword.keys(kw)} end)

    hash =
      :crypto.hash(:sha256, [ir_text, :erlang.term_to_binary({trait_summary, iface_summary})])
      |> Base.encode16(case: :lower)

    "sha256:" <> hash
  end

  @doc """
  Verifies that every operation belonging to `dialect_mod` inside `ir` satisfies the given
  layer rule (either an enumerable of allowed layers or `[allow: layers, forbid: layers]`).
  """
  @spec verify_layers(MLIR.Module.t() | MLIR.Operation.t(), module(), Enumerable.t() | keyword()) ::
          :ok | {:error, [%{operation: String.t(), layer: term()}]}
  def verify_layers(ir, dialect_mod, rule) when is_atom(dialect_mod) do
    prefix = dialect_mod.__slang_dialect_name__() <> "."
    {allow_set, forbid_set} = parse_layer_rule(rule)

    violations =
      ir
      |> Beaver.Walker.find_ops(fn op -> String.starts_with?(MLIR.Operation.name(op), prefix) end)
      |> Enum.flat_map(&layer_violation(dialect_mod, &1, allow_set, forbid_set))

    if violations == [], do: :ok, else: {:error, violations}
  end

  defp parse_layer_rule(rule) when is_list(rule) do
    if Keyword.keyword?(rule) and (Keyword.has_key?(rule, :allow) or Keyword.has_key?(rule, :forbid)) do
      {optional_map_set(Keyword.fetch(rule, :allow)), optional_map_set(Keyword.fetch(rule, :forbid)) || MapSet.new()}
    else
      {MapSet.new(rule), MapSet.new()}
    end
  end

  defp parse_layer_rule(rule), do: {MapSet.new(rule), MapSet.new()}

  defp optional_map_set({:ok, list}), do: MapSet.new(list)
  defp optional_map_set(:error), do: nil

  defp layer_violation(dialect_mod, op, allow_set, forbid_set) do
    op_name = MLIR.Operation.name(op)
    layer = dialect_mod.__slang_layer__(op_name)

    if layer_allowed?(layer, allow_set, forbid_set),
      do: [],
      else: [%{operation: op_name, layer: layer}]
  end

  defp layer_allowed?(layer, nil, forbid_set), do: not MapSet.member?(forbid_set, layer)

  defp layer_allowed?(layer, allow_set, forbid_set),
    do: MapSet.member?(allow_set, layer) and not MapSet.member?(forbid_set, layer)

  @doc "Verifies dialect operation layers via `verify_layers/3`, returning `ir` or raising `ArgumentError`."
  @spec verify_layers!(MLIR.Module.t() | MLIR.Operation.t(), module(), Enumerable.t() | keyword()) ::
          MLIR.Module.t() | MLIR.Operation.t()
  def verify_layers!(ir, dialect_mod, rule) do
    case verify_layers(ir, dialect_mod, rule) do
      :ok ->
        ir

      {:error, violations} ->
        summary =
          violations
          |> Enum.map(fn %{operation: name, layer: l} -> "#{name} (layer: #{inspect(l)})" end)
          |> Enum.uniq()
          |> Enum.join(", ")

        raise ArgumentError,
              "layer verification failed for #{inspect(dialect_mod)} (#{inspect(rule)}): #{summary}"
    end
  end

  defp expand_table_op!(entry, shared_opts) do
    {op_name, operands, results, attributes, op_opts} = unpack_table_op!(entry)
    merged_opts = merge_table_op_opts(shared_opts, op_opts)

    {operand_asts, explicit_op_names} = expand_table_items!(operands, merged_opts[:operand_names])
    {result_asts, explicit_res_names} = expand_table_items!(results, merged_opts[:result_names])

    attr_asts =
      Enum.map(attributes || [], fn {k, v} ->
        {if(is_atom(k), do: k, else: String.to_atom(to_string(k))), expand_table_constraint(v)}
      end)

    call_ast = {op_name, [], operand_asts}

    defop_opts =
      merged_opts
      |> Keyword.put(:results, result_asts)
      |> maybe_put_kw(:operand_names, explicit_op_names)
      |> maybe_put_kw(:result_names, explicit_res_names)
      |> maybe_put_kw(:attributes, if(attr_asts == [], do: merged_opts[:attributes], else: attr_asts))

    quote do
      Beaver.Slang.defop(unquote(call_ast), unquote(defop_opts))
    end
  end

  defp merge_table_op_opts(shared_opts, op_opts) do
    merged = Keyword.merge(shared_opts, op_opts)

    case {shared_opts[:metadata], op_opts[:metadata]} do
      {nil, _} ->
        merged

      {_, nil} ->
        merged

      {s_meta, o_meta} ->
        Keyword.put(
          merged,
          :metadata,
          quote(do: Map.merge(Map.new(unquote(s_meta)), Map.new(unquote(o_meta))))
        )
    end
  end

  defp maybe_put_kw(kw, _key, nil), do: kw
  defp maybe_put_kw(kw, key, val), do: Keyword.put(kw, key, val)

  defp unpack_table_op!({:{}, _meta, [name, operands, results]}),
    do: {to_op_atom!(name), operands, results, [], []}

  defp unpack_table_op!({:{}, _meta, [name, operands, results, attributes]}),
    do: {to_op_atom!(name), operands, results, attributes, []}

  defp unpack_table_op!({:{}, _meta, [name, operands, results, attributes, op_opts]}),
    do: {to_op_atom!(name), operands, results, attributes, op_opts}

  defp unpack_table_op!({:%{}, _meta, kvs}), do: unpack_table_op!(kvs)

  defp unpack_table_op!({name, operands, results}) when is_atom(name) or is_binary(name),
    do: {to_op_atom!(name), operands, results, [], []}

  defp unpack_table_op!({name, operands, results, attributes})
       when is_atom(name) or is_binary(name),
       do: {to_op_atom!(name), operands, results, attributes, []}

  defp unpack_table_op!({name, operands, results, attributes, op_opts})
       when is_atom(name) or is_binary(name),
       do: {to_op_atom!(name), operands, results, attributes, op_opts}

  defp unpack_table_op!(%{} = map), do: unpack_table_op!(Map.to_list(map))

  defp unpack_table_op!(kw) when is_list(kw) do
    name = Keyword.fetch!(kw, :name) |> to_op_atom!()
    operands = Keyword.get(kw, :operands, [])
    results = Keyword.get(kw, :results, [])
    attributes = Keyword.get(kw, :attributes, [])
    rest = Keyword.drop(kw, [:name, :operands, :results, :attributes])
    {name, operands, results, attributes, rest}
  end

  defp unpack_table_op!(other),
    do: raise(ArgumentError, "invalid Slang defops specification: #{inspect(other)}")

  defp to_op_atom!(a) when is_atom(a), do: a
  defp to_op_atom!(s) when is_binary(s) and s != "", do: String.to_atom(s)

  defp expand_table_items!(items, explicit_names) when is_list(items) do
    if items != [] and named_value_keyword?(items) do
      if explicit_names,
        do: raise(ArgumentError, "cannot combine named defops items with explicit *_names option")

      names = Enum.map(items, fn {k, _} -> to_string(k) end)
      asts = Enum.map(items, fn {_, spec} -> expand_table_constraint(spec) end)
      {asts, names}
    else
      {Enum.map(items, &expand_table_constraint/1), explicit_names}
    end
  end

  defp expand_table_constraint(:any), do: quote(do: Beaver.Slang.any())
  defp expand_table_constraint(:i1), do: quote(do: Beaver.MLIR.Type.i1())
  defp expand_table_constraint(:i8), do: quote(do: Beaver.MLIR.Type.i8())
  defp expand_table_constraint(:i16), do: quote(do: Beaver.MLIR.Type.i16())
  defp expand_table_constraint(:i32), do: quote(do: Beaver.MLIR.Type.i32())
  defp expand_table_constraint(:i64), do: quote(do: Beaver.MLIR.Type.i64())
  defp expand_table_constraint(:i128), do: quote(do: Beaver.MLIR.Type.i128())
  defp expand_table_constraint(:index), do: quote(do: Beaver.MLIR.Type.index())
  defp expand_table_constraint(:f16), do: quote(do: Beaver.MLIR.Type.f16())
  defp expand_table_constraint(:bf16), do: quote(do: Beaver.MLIR.Type.bf16())
  defp expand_table_constraint(:f32), do: quote(do: Beaver.MLIR.Type.f32())
  defp expand_table_constraint(:f64), do: quote(do: Beaver.MLIR.Type.f64())

  defp expand_table_constraint(pin_name)
       when is_atom(pin_name) and not is_boolean(pin_name) and not is_nil(pin_name) do
    var = {pin_name, [], nil}
    quote(do: ^unquote(var))
  end

  defp expand_table_constraint({:pin, pin_name}) when is_atom(pin_name) do
    var = {pin_name, [], nil}
    quote(do: ^unquote(var))
  end

  defp expand_table_constraint({:type, str}) when is_binary(str),
    do: quote(do: Beaver.MLIR.Type.get(unquote(str)))

  defp expand_table_constraint(str) when is_binary(str) do
    cond do
      String.starts_with?(str, "#") and not String.contains?(str, "<") ->
        quote(do: Beaver.Slang.base(unquote(str)))

      String.starts_with?(str, "!") and not String.contains?(str, "<") ->
        quote(do: Beaver.Slang.base(unquote(str)))

      true ->
        quote(do: Beaver.MLIR.Type.get(unquote(str)))
    end
  end

  defp expand_table_constraint({tag, inner}) when tag in [:variadic, :optional, :single] do
    expanded = expand_table_constraint(inner)
    quote(do: Beaver.Slang.unquote(tag)(unquote(expanded)))
  end

  defp expand_table_constraint({:base, inner}), do: quote(do: Beaver.Slang.base(unquote(inner)))

  defp expand_table_constraint({comb, list}) when comb in [:any_of, :all_of] and is_list(list) do
    expanded = Enum.map(list, &expand_table_constraint/1)
    quote(do: Beaver.Slang.unquote(comb)(unquote(expanded)))
  end

  defp expand_table_constraint({:is, inner}), do: quote(do: Beaver.Slang.is(unquote(inner)))
  defp expand_table_constraint(ast), do: ast

  @doc """
  This function loads the MLIR dialect into the MLIR context. It invokes the internal function of the provided module to create the dialect's IRDL module and performs additional MLIR transformations and verification.
  """
  def load(ctx, mod) when is_atom(mod) do
    dialect_name = mod.__slang_dialect_name__()

    result =
      if dynamic_dialect_loaded?(ctx, dialect_name),
        do: CAPI.beaverLogicalResultSuccess(),
        else:
          apply(mod, :__slang_dialect__, [ctx])
          |> Beaver.MLIR.Transform.canonicalize()
          |> Beaver.Composer.run!()
          |> MLIR.verify!()
          |> CAPI.mlirLoadIRDLDialects()

    if MLIR.LogicalResult.success?(result) do
      Beaver.MLIR.Trait.attach_all(ctx, dialect_name, mod.__slang_traits__())

      Beaver.MLIR.ExternalInterface.attach_all(ctx, dialect_name, mod.__slang_interfaces__())
    end

    result
  end

  defp dynamic_dialect_loaded?(ctx, dialect_name) do
    dialect =
      CAPI.mlirContextGetLoadedDialect(ctx, StringRef.create(dialect_name))

    if CAPI.beaverIsNullDialect(dialect) |> Beaver.Native.to_term() do
      false
    else
      unless CAPI.mlirDialectIsAExtensibleDialect(dialect) |> Beaver.Native.to_term() do
        raise ArgumentError,
              "cannot load Slang dialect #{inspect(dialect_name)}: " <>
                "a non-extensible dialect with that namespace is already loaded"
      end

      true
    end
  end
end
