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
    end
  end

  @doc false
  defmacro __before_compile__(_env) do
    quote do
      @doc false
      def __slang_dialect__(ctx),
        do: Beaver.Slang.create_dialect(@__slang_dialect_name__, Enum.reverse(@__slang__creator__), ctx: ctx)

      @doc false
      def __slang_traits__, do: Enum.reverse(@__slang__trait__)

      @doc false
      def __slang_interfaces__, do: Enum.reverse(@__slang__interface__)

      @doc false
      def __slang_dialect_name__, do: @__slang_dialect_name__

      use Beaver.MLIR.Dialect,
        dialect: @__slang_dialect_name__,
        ops: @__slang__operation__ || []
    end
  end

  defp creator_env do
    quote do
      [ip: Beaver.Env.block(), ctx: Beaver.Env.context(), loc: Kernel.var!(slang_internal_source_loc)]
    end
  end

  @doc false
  defp transform_defop_pins({:^, _line1, [{name, _line2, nil}]}) do
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
      region = CAPI.mlirRegionCreate()
      block = MLIR.Block.create()
      CAPI.mlirRegionAppendOwnedBlock(region, block)
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
          CAPI.mlirRegionDestroy(region)
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
      {_name, _line0, nil} ->
        nil

      {:=, _line0, [_var, _right]} ->
        ast

      {variadic_tag, {_name, _line0, nil}} when variadic_tag in @variadic_tags ->
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
      {_name, _line0, nil} -> ast
      {:=, _line0, [var, _right]} -> var
      {variadic_tag, {_name, _line0, nil}} when variadic_tag in @variadic_tags -> ast
      _ -> get_slang_arg_ast(i)
    end
  end

  defp get_args_as_vars(args) do
    for {v, i} <- Enum.with_index(args), do: transform_variable(v, i)
  end

  defp declaration_name({:=, _, [var, _]}, index, prefix), do: declaration_name(var, index, prefix)

  defp declaration_name({tag, value}, index, prefix) when tag in @variadic_tags,
    do: declaration_name(value, index, prefix)

  defp declaration_name({name, _, nil}, _index, _prefix) when is_atom(name), do: Atom.to_string(name)

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
      |> Keyword.drop(~w(argument_names result_names attributes regions traits interfaces)a)
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
      Keyword.validate!(block || [], ~w(do results result_names operand_names attributes regions traits interfaces)a)

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
      source: %{file: __CALLER__.file, line: __CALLER__.line}
    )
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
