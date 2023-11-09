defmodule Beaver.Slang do
  use Beaver
  alias Beaver.MLIR.Dialect.IRDL
  @variadic_tags [:variadic, :optional, :single]
  @callback __slang_dialect__(ctx :: Beaver.MLIR.Context.t()) :: :ok | {:error, String.t()}
  @moduledoc """
  Defining a MLIR dialect with macros in Elixir. Internally expressions are compiled to [IRDL](https://mlir.llvm.org/docs/Dialects/IRDL/)
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
    end
  end

  @doc """
  This macro is invoked before the module is compiled. Internally it defines a function which creates the MLIR dialect's IRDL module. It also uses the `Beaver.MLIR.Dialect` module to define the dialect's operations.
  """
  defmacro __before_compile__(_env) do
    quote do
      @doc false
      def __slang_dialect__(ctx) do
        Beaver.Slang.create_dialect(@__slang_dialect_name__, Enum.reverse(@__slang__creator__),
          ctx: ctx
        )
      end

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
        opts = [ctx: Beaver.Env.context(), block: Beaver.Env.block()]
        unquote(alias_name)(opts)
      end
    end
  end

  defp transform_defop_pins({:=, _line0, [_var, {:=, _line1, _right2}]} = ast), do: ast
  # only leaf assignment should be transformed
  defp transform_defop_pins({:=, _line0, [var, right]}) do
    quote do
      unquote(var) =
        Beaver.Slang.create_constrain(unquote(right),
          block: Beaver.Env.block(),
          ctx: Beaver.Env.context()
        )
    end
  end

  defp transform_defop_pins(ast), do: ast

  @doc false
  # This function creates a `irdl.is` op for a given value. It uses the mlir macro to generate the MLIR code for the constrain attribute.
  def create_constrain({variadic_tag, v}, opts) when variadic_tag in @variadic_tags do
    {variadic_tag, create_constrain(v, opts)}
  end

  def create_constrain(%MLIR.Value{} = v, _opts), do: v

  def create_constrain(t, opts) do
    use Beaver

    mlir ctx: opts[:ctx], block: opts[:block] do
      Beaver.MLIR.Dialect.IRDL.is(expected: t) >>> ~t{!irdl.attribute}
    end
  end

  @doc false
  # This function applies the target op to the given SSA (Static Single Assignment) form.
  defp op_applier(ssa) do
    i =
      Enum.find_index(ssa.arguments, fn
        {:slang_target_op, _} -> true
        _ -> false
      end)

    {{:slang_target_op, op}, arguments} = List.pop_at(ssa.arguments, i)
    apply(Beaver.MLIR.Dialect.IRDL, op, [%{ssa | arguments: arguments}])
  end

  # This function determines the variadicity of the given values based on the provided options. It generates the variadicity attribute for the values if needed.
  defp get_variadicity(values, opts) do
    use Beaver

    if opts[:need_variadicity] do
      tags =
        values
        |> List.wrap()
        |> Enum.map(fn
          {variadic_tag, _} when variadic_tag in @variadic_tags ->
            variadic_tag |> Atom.to_string()

          _ ->
            "single"
        end)
        |> Enum.join(",")

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
        mlir block: opts[:block], ctx: ctx do
          op_applier slang_target_op: op, sym_name: "\"#{name}\"" do
            region do
              block b_op() do
                {args, ret} = constrain_f.(block: Beaver.Env.block(), ctx: ctx)

                case strip_variadicity(args) do
                  [] ->
                    []

                  args ->
                    op_applier(
                      args,
                      get_variadicity(args, opts),
                      slang_target_op: args_op
                    ) >>> []
                end

                case {return_op, strip_variadicity(ret)} do
                  {_, []} ->
                    []

                  {return_op, ret} when not is_nil(return_op) ->
                    op_applier(
                      ret
                      |> Enum.map(&create_constrain(&1, block: Beaver.Env.block(), ctx: ctx)),
                      get_variadicity(ret, opts),
                      slang_target_op: return_op
                    ) >>> []
                end
              end
            end
          end >>> []
        end
      end
    )
  end

  # This function generates the AST for an argument based on the given index.
  defp get_slang_arg_ast(i) do
    {"slang_internal_arg#{i}" |> String.to_atom(), [], nil}
  end

  # This function transforms the given argument AST based on the provided usage (if it is using as variable or constrain declaration). It handles different cases based on the structure of the argument and returns the transformed AST.
  defp transform_arg(ast, i, usage) when usage in [:constrain, :variable] do
    case ast do
      {_name, _line0, nil} ->
        case usage do
          # erase hanging vars to suppress warning
          :constrain ->
            nil

          :variable ->
            ast
        end

      {:=, _line0, [var, _right]} ->
        case usage do
          :constrain ->
            ast

          :variable ->
            var
        end

      {variadic_tag, {_name, _line0, nil}}
      when variadic_tag in @variadic_tags ->
        ast

      _ ->
        case usage do
          :constrain ->
            quote do
              unquote(get_slang_arg_ast(i)) =
                Beaver.Slang.create_constrain(unquote(ast),
                  block: Beaver.Env.block(),
                  ctx: Beaver.Env.context()
                )
            end

          :variable ->
            get_slang_arg_ast(i)
        end
    end
  end

  # This function generates the AST for the arguments of a creator function as variables.
  defp get_args_as_vars(args) do
    for {v, i} <- Enum.with_index(args), do: transform_arg(v, i, :variable)
  end

  # This function generates the AST for a creator function for an IRDL operation (like `irdl.operation`, `irdl.type`). It uses the transform_defop_pins/1 function to transform the pins, generates the MLIR code for the operation and its arguments, and applies the operation using op_applier/1.
  defp gen_creator(op, args_op, call, block, opts \\ []) do
    {name, args} = call |> Macro.decompose_call()
    name = Atom.to_string(name)
    creator = String.to_atom("create_" <> name)
    attr_name = String.to_atom("__slang__" <> "#{op}" <> "__")
    args_var_ast = get_args_as_vars(args)

    input_constrains =
      args
      |> Macro.postwalk(&transform_defop_pins/1)
      |> Enum.with_index()
      |> Enum.map(fn {ast, i} ->
        transform_arg(ast, i, :constrain)
      end)

    quote do
      Module.put_attribute(__MODULE__, unquote(attr_name), unquote(name))
      @__slang__creator__ {unquote(op), __MODULE__, unquote(creator)}
      def unquote(creator)(opts) do
        Beaver.Slang.run_creator(
          unquote(name),
          unquote(op),
          unquote(args_op),
          fn opts ->
            use Beaver

            mlir block: opts[:block], ctx: opts[:ctx] do
              unquote_splicing(input_constrains)
              {[unquote_splicing(args_var_ast)], unquote(block[:do])}
            end
          end,
          opts ++ unquote(opts)
        )
      end
    end
  end

  defp create_element(element, dialect, name, params, opts \\ [])
       when element in [:type, :attr] do
    params = %MLIR.Attribute{} = MLIR.Attribute.array(params, opts)

    apply(
      MLIR.CAPI,
      case element do
        :attr ->
          :beaverGetIRDLDefinedAttr

        :type ->
          :beaverGetIRDLDefinedType
      end,
      [dialect, name, params]
    )
  end

  @doc """
  This macro defines a type in the dialect. It generates the AST for the creator function and the definition function for the type.
  """
  defmacro deftype(call, block \\ nil) do
    {name, args} =
      call
      |> Macro.decompose_call()

    quote do
      unquote(gen_creator(:type, :parameters, call, block, need_variadicity: false))

      def unquote(name)(unquote_splicing(get_args_as_vars(args))) do
        {:parametric, unquote(name), [unquote_splicing(get_args_as_vars(args))]}
      end
    end
  end

  @doc """
  This macro defines an operation in the dialect. It generates the AST for the creator function for the operation.
  """
  defmacro defop(call, block \\ nil) do
    gen_creator(:operation, :operands, call, block, return_op: :results, need_variadicity: true)
  end

  defp get_alias_name(def_name) do
    String.to_atom("alias_" <> Atom.to_string(def_name))
  end

  @doc """
  This macro defines an alias for a lengthy type. It generates the AST for the alias function.
  """
  defmacro defalias(call, block) do
    {name, _args} = call |> Macro.decompose_call()
    alias_name = get_alias_name(name)
    block = block |> Macro.postwalk(&transform_defop_pins/1)

    quote do
      @slang_alias unquote(alias_name)
      def unquote(alias_name)(opts) do
        use Beaver

        mlir ctx: opts[:ctx], block: opts[:block] do
          unquote(block[:do])
          |> Beaver.Slang.create_parametric(
            ctx: Beaver.Env.context(),
            block: Beaver.Env.block()
          )
        end
      end
    end
  end

  @doc """
  This macro generates the AST for the any_of attribute in the dialect.
  """
  defmacro any_of(types) do
    quote do
      use Beaver

      mlir do
        types =
          for t <- unquote(types) do
            Beaver.MLIR.Dialect.IRDL.is(expected: t) >>> ~t{!irdl.attribute}
          end

        Beaver.MLIR.Dialect.IRDL.any_of(types) >>> ~t{!irdl.attribute}
      end
    end
  end

  @doc """
  This macro generates the AST for `irdl.is` op, usually used to create a constrain on type
  """
  defmacro is(type) do
    quote do
      use Beaver

      mlir do
        Beaver.MLIR.Dialect.IRDL.is(expected: unquote(type)) >>>
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
        Beaver.MLIR.Dialect.IRDL.any() >>> ~t{!irdl.attribute}
      end
    end
  end

  @doc false
  # This function creates a parametric attribute for a given value. It generates the code for `irdl.parametric` op.
  def create_parametric({:parametric, symbol, values}, opts) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        mlir ctx: ctx, block: opts[:block] do
          IRDL.parametric(values,
            base_type: MLIR.Attribute.flat_symbol_ref("#{symbol}", ctx: Beaver.Env.context())
          ) >>> ~t{!irdl.attribute}
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
                block b_dialect() do
                  for {_type, m, f} <- creators do
                    opts = [ctx: Beaver.Env.context(), block: Beaver.Env.block()]
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

  @doc """
  This function loads the MLIR dialect into the MLIR context. It invokes the internal function of the provided module to create the dialect's IRDL module and performs additional MLIR transformations and verification.
  """
  def load(ctx, mod) when is_atom(mod) do
    apply(mod, :__slang_dialect__, [ctx])
    |> Beaver.MLIR.Transforms.canonicalize()
    |> MLIR.Pass.Composer.run!()
    |> MLIR.Operation.verify!()
    |> Beaver.MLIR.CAPI.beaverLoadIRDLDialects()
  end
end
