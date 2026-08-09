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
  built-in and callback-backed custom operation traits can be attached when
  the dialect is loaded.

  Schema construction and runtime interface attachment are separate:
  `__slang_dialect__/1` builds the inspectable IRDL module, while `load/2`
  verifies and registers that module before attaching traits. See the
  [Slang guide](slang.html) for a complete dialect and the supported syntax.
  """

  @doc """
  This macro is invoked when the module is used. It sets up the module by registering attributes and importing macros from `Beaver.Slang`.
  """
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
  # This macro is invoked before the module is compiled. Internally it defines a function which creates the MLIR dialect's IRDL module. It also uses the `Beaver.MLIR.Dialect` module to define the dialect's operations.
  defmacro __before_compile__(_env) do
    quote do
      @doc false
      def __slang_dialect__(ctx) do
        Beaver.Slang.create_dialect(@__slang_dialect_name__, Enum.reverse(@__slang__creator__),
          ctx: ctx
        )
      end

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

  @doc false
  # This function transforms the `^argument`s of a defop macro call. It handles different cases based on the structure of the pins and returns the transformed AST.
  defp transform_defop_pins({:^, _line1, [{name, _line2, nil}]}) do
    alias_name = get_alias_name(name)

    quote do
      mlir do
        opts = [
          ctx: Beaver.Env.context(),
          ip: Beaver.Env.block(),
          loc: Kernel.var!(slang_internal_source_loc)
        ]

        unquote(alias_name)(opts)
      end
    end
  end

  defp transform_defop_pins({:=, _line0, [_var, {:=, _line1, _right2}]} = ast), do: ast
  # only leaf assignment should be transformed
  defp transform_defop_pins({:=, _line0, [var, right]}) do
    quote do
      unquote(var) =
        Beaver.Slang.create_constraint(unquote(right),
          ip: Beaver.Env.block(),
          ctx: Beaver.Env.context(),
          loc: Kernel.var!(slang_internal_source_loc)
        )
    end
  end

  defp transform_defop_pins(ast), do: ast

  @doc false
  # This function creates a `irdl.is` op for a given value. It uses the mlir macro to generate the MLIR code for the constraint attribute.
  def create_constraint({variadic_tag, v}, opts) when variadic_tag in @variadic_tags do
    {variadic_tag, create_constraint(v, opts)}
  end

  def create_constraint(%MLIR.Value{} = v, _opts), do: v

  def create_constraint(t, opts) do
    use Beaver

    mlir ctx: opts[:ctx], ip: Beaver.Deferred.fetch_insertion_point(opts) do
      loc = constraint_location(opts, Beaver.Env.context())
      Beaver.MLIR.Dialect.IRDL.is([loc, expected: t]) >>> ~t{!irdl.attribute}
    end
  end

  @doc false
  # This function applies the target op to the given SSA (Static Single Assignment) form.
  defp op_applier(ssa) do
    {op, arguments} = pop_metadata!(ssa.arguments, :slang_target_op)
    {names, arguments} = pop_metadata!(arguments, :slang_names)
    {loc, arguments} = pop_metadata!(arguments, :slang_location)
    n = arguments |> Enum.count(&match?(%MLIR.Value{}, &1))

    if length(names) != n do
      raise ArgumentError,
            "#{op} declares #{length(names)} names for #{n} constraints: #{inspect(names)}"
    end

    names = Enum.map(names, &MLIR.Attribute.string/1)

    names_attr =
      case op do
        op when op in [:operands, :results, :parameters, :regions] ->
          [names: MLIR.Attribute.array(names)]

        :attributes ->
          [attributeValueNames: MLIR.Attribute.array(names)]

        _ ->
          []
      end

    arguments = arguments ++ names_attr

    apply(Beaver.MLIR.Dialect.IRDL, op, [
      %{ssa | arguments: arguments, loc: loc}
    ])
  end

  defp pop_metadata!(arguments, key) do
    case Enum.find_index(arguments, &match?({^key, _}, &1)) do
      nil ->
        raise ArgumentError, "missing internal Slang metadata #{inspect(key)}"

      index ->
        {{^key, value}, arguments} = List.pop_at(arguments, index)
        {value, arguments}
    end
  end

  # This function determines the variadicity of the given values based on the provided options. It generates the variadicity attribute for the values if needed.
  defp get_variadicity(values, opts) do
    use Beaver

    if opts[:need_variadicity] do
      tags =
        values
        |> List.wrap()
        |> Enum.map_join(",", fn
          {variadic_tag, _} when variadic_tag in @variadic_tags ->
            variadic_tag |> Atom.to_string()

          _ ->
            "single"
        end)

      [
        variadicity: ~a{#irdl<variadicity_array[#{tags}]>}
      ]
    else
      []
    end
  end

  # This function removes the variadicity tags from the given values.
  defp strip_variadicity(values) do
    values
    |> List.wrap()
    |> Enum.map(fn
      {variadic_tag, v} when variadic_tag in @variadic_tags ->
        v

      v ->
        v
    end)
  end

  @doc false
  # This function runs the creator function for a given operation. It generates the MLIR code for the operation and its arguments, applies the operation using op_applier/1, and returns the result.
  def run_creator(name, op, args_op, constrain_f, opts) do
    use Beaver
    return_op = opts[:return_op]

    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        mlir ip: Beaver.Deferred.fetch_insertion_point(opts), ctx: ctx do
          source_loc = source_location(opts, ctx)

          op_applier slang_target_op: op,
                     slang_names: [],
                     slang_location: source_loc,
                     sym_name: "\"#{name}\"" do
            region do
              block _op() do
                {args, ret, attributes, regions} = constrain_f.(Beaver.Env.block(), ctx)

                case {args, strip_variadicity(args)} do
                  {_, []} ->
                    []

                  {descriptors, args} ->
                    op_applier(
                      args,
                      get_variadicity(descriptors, opts),
                      slang_target_op: args_op,
                      slang_names: opts[:argument_names],
                      slang_location: source_loc
                    ) >>> []
                end

                case {return_op, ret, strip_variadicity(ret)} do
                  {_, _, []} ->
                    []

                  {return_op, descriptors, ret} when not is_nil(return_op) ->
                    op_applier(
                      ret
                      |> Enum.map(
                        &create_constraint(&1,
                          ip: Beaver.Env.block(),
                          ctx: ctx,
                          loc: source_loc
                        )
                      ),
                      get_variadicity(descriptors, opts),
                      slang_target_op: return_op,
                      slang_names: opts[:result_names],
                      slang_location: source_loc
                    ) >>> []
                end

                case attributes do
                  [] ->
                    []

                  attributes ->
                    op_applier(
                      attributes,
                      slang_target_op: :attributes,
                      slang_names: opts[:attribute_names],
                      slang_location: source_loc
                    ) >>> []
                end

                case regions do
                  [] ->
                    []

                  regions ->
                    op_applier(
                      regions,
                      slang_target_op: :regions,
                      slang_names: opts[:region_names],
                      slang_location: source_loc
                    ) >>> []
                end
              end
            end
          end >>> []
        end
      end
    )
  end

  defp source_location(opts, ctx) do
    case opts[:source] do
      %{file: file, line: line} ->
        Beaver.Deferred.create(MLIR.Location.file(name: file, line: line), ctx)

      _ ->
        Beaver.Deferred.create(MLIR.Location.unknown(), ctx)
    end
  end

  @doc false
  def constraint_location(opts, ctx) do
    opts[:loc] || source_location(opts, ctx)
  end

  # This function generates the AST for an argument based on the given index.
  defp get_slang_arg_ast(i) do
    {"slang_internal_arg#{i}" |> String.to_atom(), [], nil}
  end

  # This function transforms the given argument AST based on the provided usage (if it is using as variable or constraint declaration). It handles different cases based on the structure of the argument and returns the transformed AST.
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
            Beaver.Slang.create_constraint(unquote(ast),
              ip: Beaver.Env.block(),
              ctx: Beaver.Env.context(),
              loc: Kernel.var!(slang_internal_source_loc)
            )
        end
    end
  end

  defp transform_variable(ast, i) do
    case ast do
      {_name, _line0, nil} ->
        ast

      {:=, _line0, [var, _right]} ->
        var

      {variadic_tag, {_name, _line0, nil}} when variadic_tag in @variadic_tags ->
        ast

      _ ->
        get_slang_arg_ast(i)
    end
  end

  # This function generates the AST for the arguments of a creator function as variables.
  defp get_args_as_vars(args) do
    for {v, i} <- Enum.with_index(args), do: transform_variable(v, i)
  end

  defp declaration_name({:=, _, [var, _]}, index, prefix),
    do: declaration_name(var, index, prefix)

  defp declaration_name({tag, value}, index, prefix) when tag in @variadic_tags,
    do: declaration_name(value, index, prefix)

  defp declaration_name({name, _, nil}, _index, _prefix) when is_atom(name),
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

    if length(asts) != length(names) do
      raise ArgumentError,
            "expected #{length(asts)} names, got #{length(names)}: #{inspect(names)}"
    end

    if length(Enum.uniq(names)) != length(names) do
      raise ArgumentError, "Slang declaration names must be unique: #{inspect(names)}"
    end

    names
  end

  defp normalize_names!(_asts, names, _prefix) do
    raise ArgumentError, "expected a list of Slang declaration names, got: #{inspect(names)}"
  end

  defp normalize_name!(name) when is_atom(name), do: Atom.to_string(name)
  defp normalize_name!(name) when is_binary(name) and name != "", do: name

  defp normalize_name!(name) do
    raise ArgumentError, "expected a non-empty atom or string name, got: #{inspect(name)}"
  end

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

    if MapSet.member?(used, candidate) do
      available_name(name, used, suffix + 1)
    else
      candidate
    end
  end

  defp uniquify_declaration_names(argument_names, result_names, attribute_names, region_names) do
    names = uniquify_names(argument_names ++ result_names ++ attribute_names ++ region_names)

    {argument_names, names} = Enum.split(names, length(argument_names))
    {result_names, names} = Enum.split(names, length(result_names))
    {attribute_names, region_names} = Enum.split(names, length(attribute_names))

    {argument_names, result_names, attribute_names, region_names}
  end

  defp split_named_values(nil, _prefix, _explicit_names), do: {[], []}

  defp split_named_values(values, prefix, explicit_names) when is_list(values) do
    if named_value_keyword?(values) do
      if explicit_names do
        raise ArgumentError,
              "cannot combine named #{prefix} entries with an explicit #{prefix}_names option"
      end

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
    Keyword.keyword?(values) and
      Enum.all?(values, fn {name, _value} -> name not in @variadic_tags end)
  end

  defp normalize_results_ast(nil, _explicit_names), do: {quote(do: []), []}

  defp normalize_results_ast({:__block__, meta, expressions}, explicit_names) do
    {result_values, result_names} =
      expressions
      |> List.last()
      |> split_named_values("result", explicit_names)

    result_expression = quote(do: [unquote_splicing(result_values)])
    expressions = List.replace_at(expressions, -1, result_expression)
    {{:__block__, meta, expressions}, result_names}
  end

  defp normalize_results_ast(results, explicit_names) do
    {result_values, result_names} = split_named_values(results, "result", explicit_names)
    {quote(do: [unquote_splicing(result_values)]), result_names}
  end

  defp split_attributes(nil), do: {[], []}

  defp split_attributes(attributes) when is_list(attributes) do
    if Keyword.keyword?(attributes) do
      {Enum.map(attributes, &elem(&1, 1)),
       Enum.map(attributes, &(&1 |> elem(0) |> normalize_name!()))}
    else
      raise ArgumentError,
            "operation attributes must be a keyword list of name: constraint entries"
    end
  end

  defp split_attributes(attributes) do
    raise ArgumentError,
          "operation attributes must be a keyword list, got: #{Macro.to_string(attributes)}"
  end

  defp normalize_region_descriptor(:any) do
    %{args: nil, size: nil}
  end

  defp normalize_region_descriptor({:sized, size}) when is_integer(size) and size > 0 do
    %{args: nil, size: size}
  end

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
      []
      |> then(fn attrs ->
        if is_nil(size) do
          attrs
        else
          attrs ++
            [
              numberOfBlocks:
                Beaver.Deferred.create(
                  MLIR.Attribute.integer(MLIR.Type.i32(), size),
                  ctx
                )
            ]
        end
      end)
      |> then(fn attrs ->
        if constrained_args? do
          attrs ++ [constrainedArguments: Beaver.Deferred.create(MLIR.Attribute.unit(), ctx)]
        else
          attrs
        end
      end)

    args ++ attrs
  end

  defp do_gen_region(region, block, ctx, loc) do
    use Beaver

    region_args =
      region
      |> normalize_region_descriptor()
      |> build_region_arguments(block, ctx, loc)

    %Beaver.SSA{
      arguments: region_args,
      results: [Beaver.Deferred.create(~t{!irdl.region}, ctx)],
      ip: block,
      ctx: ctx,
      loc: loc,
      evaluator: &MLIR.Operation.eval_ssa/1
    }
    |> IRDL.region()
  end

  @doc false
  def gen_region(regions, block, ctx, loc) do
    regions
    |> List.wrap()
    |> Enum.map(&do_gen_region(&1, block, ctx, loc))
  end

  @doc false
  # This function generates the AST for a creator function for an IRDL operation (like `irdl.operation`, `irdl.type`). It uses the transform_defop_pins/1 function to transform the pins, generates the MLIR code for the operation and its arguments, and applies the operation using op_applier/1.
  defp gen_creator(op, args_op, call, do_block, opts) do
    {name, args} = call |> Macro.decompose_call()
    name = Atom.to_string(name)
    creator = String.to_atom("create_" <> name)
    attr_name = String.to_atom("__slang__" <> "#{op}" <> "__")
    args_var_ast = get_args_as_vars(args)
    argument_names = normalize_names!(args, opts[:argument_names], Atom.to_string(args_op))

    {result_ast, result_names} = normalize_results_ast(do_block, opts[:result_names])
    {attribute_values, attribute_names} = split_attributes(opts[:attributes])
    {region_values, region_names} = split_named_values(opts[:regions], "region", nil)

    {argument_names, result_names, attribute_names, region_names} =
      uniquify_declaration_names(
        argument_names,
        result_names,
        attribute_names,
        region_names
      )

    input_constrains =
      args
      |> Macro.postwalk(&transform_defop_pins/1)
      |> Enum.with_index()
      |> Enum.map(fn {ast, i} ->
        transform_constraint(ast, i)
      end)

    {attribute_constraints, attribute_vars} =
      attribute_values
      |> Macro.postwalk(&transform_defop_pins/1)
      |> Enum.with_index(length(args))
      |> Enum.map_reduce([], fn {ast, i}, vars ->
        {transform_constraint(ast, i), vars ++ [transform_variable(ast, i)]}
      end)

    creator_opts =
      opts
      |> Keyword.drop([
        :argument_names,
        :result_names,
        :attributes,
        :regions,
        :traits,
        :interfaces
      ])
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
      unquote(
        if traits == [] do
          quote(do: :ok)
        else
          quote do
            @__slang__trait__ {unquote(name), unquote(traits)}
          end
        end
      )

      unquote(
        if interfaces == [] do
          quote(do: :ok)
        else
          quote do
            @__slang__interface__ {unquote(name), unquote(interfaces)}
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
                Beaver.Slang.gen_region(
                  unquote(region_values),
                  block,
                  ctx,
                  Kernel.var!(slang_internal_source_loc)
                )

              {
                [unquote_splicing(args_var_ast)],
                unquote(result_ast),
                [unquote_splicing(attribute_vars)],
                regions
              }
            end
          end,
          opts ++ unquote(Macro.escape(creator_opts))
        )
      end
    end
  end

  @doc false
  def create_constrained_element(element, dialect, name, params, opts \\ []) do
    Beaver.Deferred.from_opts(opts, fn ctx ->
      params =
        params
        |> Enum.map(&Beaver.Deferred.create(&1, ctx))
        |> Enum.map(fn
          %MLIR.Type{} = t -> MLIR.Attribute.type(t)
          %MLIR.Attribute{} = a -> a
        end)
        |> MLIR.Attribute.array(ctx: ctx)

      apply(
        MLIR.CAPI,
        case element do
          :attribute ->
            :beaverIRDLGetDefinedAttr

          :type ->
            :beaverIRDLGetDefinedType
        end,
        [MLIR.StringRef.create(dialect), MLIR.StringRef.create(name), params]
      )
    end)
  end

  defp gen_create_element_creator(element, call, opts, caller) do
    {name, args} =
      call
      |> Macro.decompose_call()

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
          {:parametric,
           MLIR.Attribute.symbol_ref(@__slang_dialect_name__, [to_string(unquote(name))]),
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

  @doc """
  This macro defines a type in the dialect.
  """
  defmacro deftype(call, opts \\ []) do
    gen_create_element_creator(:type, call, opts, __CALLER__)
  end

  @doc """
  This macro defines a attribute in the dialect.
  """
  defmacro defattr(call, opts \\ []) do
    gen_create_element_creator(:attribute, call, opts, __CALLER__)
  end

  @doc """
  This macro defines an operation in the dialect. It generates the AST for the creator function for the operation.
  """
  defmacro defop(call, block \\ nil) do
    block =
      Keyword.validate!(block || [], [
        :do,
        :results,
        :result_names,
        :operand_names,
        :attributes,
        :regions,
        :traits,
        :interfaces
      ])

    if Keyword.has_key?(block, :do) and Keyword.has_key?(block, :results) do
      raise ArgumentError, "defop accepts either a do block or :results, not both"
    end

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

  defp get_alias_name(def_name) do
    String.to_atom("alias_" <> Atom.to_string(def_name))
  end

  defp gen_constraint(call, block) do
    {name, args} = call |> Macro.decompose_call()

    if args != [] do
      raise ArgumentError, "named Slang constraints do not accept arguments"
    end

    alias_name = get_alias_name(name)
    block = block |> Macro.postwalk(&transform_defop_pins/1)

    quote do
      @slang_alias unquote(alias_name)
      def unquote(alias_name)(opts) do
        use Beaver

        mlir ctx: opts[:ctx], ip: Beaver.Deferred.fetch_insertion_point(opts) do
          Kernel.var!(slang_internal_source_loc) =
            Beaver.Slang.constraint_location(opts, Beaver.Env.context())

          unquote(block[:do])
          |> Beaver.Slang.create_parametric(
            ctx: Beaver.Env.context(),
            ip: Beaver.Env.block(),
            loc: Kernel.var!(slang_internal_source_loc)
          )
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

  @doc """
  This macro generates the AST for the any_of attribute in the dialect.
  """
  defmacro any_of(types) do
    quote do
      use Beaver

      mlir do
        types =
          for t <- unquote(types) do
            Beaver.Slang.create_constraint(t,
              ip: Beaver.Env.block(),
              ctx: Beaver.Env.context(),
              loc: Kernel.var!(slang_internal_source_loc)
            )
          end

        Beaver.MLIR.Dialect.IRDL.any_of([Kernel.var!(slang_internal_source_loc) | types]) >>>
          ~t{!irdl.attribute}
      end
    end
  end

  @doc """
  Generates an `irdl.all_of` constraint.
  """
  defmacro all_of(types) do
    quote do
      use Beaver

      mlir do
        types =
          for t <- unquote(types) do
            Beaver.Slang.create_constraint(t,
              ip: Beaver.Env.block(),
              ctx: Beaver.Env.context(),
              loc: Kernel.var!(slang_internal_source_loc)
            )
          end

        Beaver.MLIR.Dialect.IRDL.all_of([Kernel.var!(slang_internal_source_loc) | types]) >>>
          ~t{!irdl.attribute}
      end
    end
  end

  @doc false
  def create_base({:parametric, symbol, _values, _element}, opts) do
    create_base({:base_ref, symbol}, opts)
  end

  def create_base({:base_ref, symbol}, opts) do
    Beaver.Deferred.from_opts(opts, fn ctx ->
      mlir ctx: ctx, ip: Beaver.Deferred.fetch_insertion_point(opts) do
        loc = constraint_location(opts, ctx)

        IRDL.base([loc, base_ref: Beaver.Deferred.create(symbol, ctx)]) >>>
          ~t{!irdl.attribute}
      end
    end)
  end

  def create_base(name, opts) when is_binary(name) do
    Beaver.Deferred.from_opts(opts, fn ctx ->
      mlir ctx: ctx, ip: Beaver.Deferred.fetch_insertion_point(opts) do
        loc = constraint_location(opts, ctx)
        IRDL.base([loc, base_name: MLIR.Attribute.string(name)]) >>> ~t{!irdl.attribute}
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
      Beaver.Slang.create_base(unquote(base),
        ip: Beaver.Env.block(),
        ctx: Beaver.Env.context(),
        loc: Kernel.var!(slang_internal_source_loc)
      )
    end
  end

  @doc "Marks an operand or result constraint as variadic."
  defmacro variadic(constraint), do: quote(do: {:variadic, unquote(constraint)})

  @doc "Marks an operand or result constraint as optional."
  defmacro optional(constraint), do: quote(do: {:optional, unquote(constraint)})

  @doc "Marks an operand or result constraint as a required singleton."
  defmacro single(constraint), do: quote(do: {:single, unquote(constraint)})

  @doc """
  This macro generates the AST for `irdl.is` op, usually used to create a constraint on type
  """
  defmacro is(type) do
    quote do
      use Beaver

      mlir do
        Beaver.MLIR.Dialect.IRDL.is([
          Kernel.var!(slang_internal_source_loc),
          expected: unquote(type)
        ]) >>>
          ~t{!irdl.attribute}
      end
    end
  end

  @doc """
  This macro generates the AST for the `irdl.any` op.
  """
  defmacro any() do
    quote do
      use Beaver

      mlir do
        Beaver.MLIR.Dialect.IRDL.any(Kernel.var!(slang_internal_source_loc)) >>>
          ~t{!irdl.attribute}
      end
    end
  end

  @doc false
  # This function creates a parametric attribute for a given value. It generates the code for `irdl.parametric` op.
  def create_parametric({:parametric, symbol, values, _}, opts) do
    base_type = Beaver.Deferred.from_opts(opts, symbol)

    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        mlir ctx: ctx, ip: Beaver.Deferred.fetch_insertion_point(opts) do
          loc = constraint_location(opts, ctx)
          IRDL.parametric([loc | values], base_type: base_type) >>> ~t{!irdl.attribute}
        end
      end
    )
  end

  @doc false
  def create_parametric(v, _opts), do: v

  @doc false
  # This function creates the MLIR dialect using the provided name and creators. It generates the MLIR code for the dialect and call all the generated creators.
  def create_dialect(name, creators, opts) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        mlir ctx: ctx do
          module do
            IRDL.dialect sym_name: "\"#{name}\"" do
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

  @external_interfaces [
    :memory_effects,
    :conditionally_speculatable,
    :transform_op,
    :pattern_descriptor
  ]

  defp normalize_interfaces!(nil), do: []

  defp normalize_interfaces!(interfaces) when is_list(interfaces) do
    unless Keyword.keyword?(interfaces) do
      raise ArgumentError, ":interfaces must be a keyword list"
    end

    unsupported = Keyword.keys(interfaces) -- @external_interfaces

    if unsupported != [] do
      raise ArgumentError, "unsupported Slang interfaces: #{inspect(unsupported)}"
    end

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
      if dynamic_dialect_loaded?(ctx, dialect_name) do
        MLIR.CAPI.mlirLogicalResultSuccess()
      else
        apply(mod, :__slang_dialect__, [ctx])
        |> Beaver.MLIR.Transform.canonicalize()
        |> Beaver.Composer.run!()
        |> MLIR.verify!()
        |> Beaver.MLIR.CAPI.mlirLoadIRDLDialects()
      end

    if MLIR.LogicalResult.success?(result) do
      Beaver.MLIR.Trait.attach_all(
        ctx,
        dialect_name,
        mod.__slang_traits__()
      )

      Beaver.MLIR.ExternalInterface.attach_all(
        ctx,
        dialect_name,
        mod.__slang_interfaces__()
      )
    end

    result
  end

  defp dynamic_dialect_loaded?(ctx, dialect_name) do
    dialect =
      MLIR.CAPI.mlirContextGetLoadedDialect(ctx, MLIR.StringRef.create(dialect_name))

    if MLIR.CAPI.beaverIsNullDialect(dialect) |> Beaver.Native.to_term() do
      false
    else
      unless MLIR.CAPI.mlirDialectIsAExtensibleDialect(dialect) |> Beaver.Native.to_term() do
        raise ArgumentError,
              "cannot load Slang dialect #{inspect(dialect_name)}: " <>
                "a non-extensible dialect with that namespace is already loaded"
      end

      true
    end
  end
end
