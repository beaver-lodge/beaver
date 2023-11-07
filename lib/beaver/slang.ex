defmodule Beaver.Slang do
  use Beaver
  alias Beaver.MLIR.Dialect.IRDL

  @callback __slang_dialect__(ctx :: Beaver.MLIR.Context.t()) :: :ok | {:error, String.t()}
  @moduledoc """
  Defining a MLIR dialect with macros in Elixir. Internally expressions are compiled to [IRDL](https://mlir.llvm.org/docs/Dialects/IRDL/)
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
  # transform pin to alias function call
  defp transform_defop_pins({:^, _line1, [{name, _line2, nil}]}) do
    alias_name = get_alias_name(name)

    quote do
      mlir do
        opts = [ctx: Beaver.Env.context(), block: Beaver.Env.block()]
        unquote(alias_name)(opts)
      end
    end
  end

  defp transform_defop_pins(ast), do: ast

  defp get_args_as_vars(args) do
    for v <- args do
      case v do
        {_name, _line0, nil} ->
          v

        {:=, _line0, [var, _right]} ->
          var

        {variadic_tag, {_name, _line0, nil}} when variadic_tag in [:variadic, :optional] ->
          v
      end
    end
  end

  @doc false
  defp op_applier(ssa) do
    i =
      Enum.find_index(ssa.arguments, fn
        {:slang_target_op, _} -> true
        _ -> false
      end)

    {{:slang_target_op, op}, arguments} = List.pop_at(ssa.arguments, i)
    apply(Beaver.MLIR.Dialect.IRDL, op, [%{ssa | arguments: arguments}])
  end

  defp get_variadicity(values, opts) do
    use Beaver

    if opts[:need_variadicity] do
      tags =
        values
        |> List.wrap()
        |> Enum.map(fn
          {:optional, _} ->
            "optional"

          {:variadic, _} ->
            "variadic"

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

  defp strip_variadicity(values) do
    values
    |> List.wrap()
    |> Enum.map(fn
      {:optional, v} ->
        v

      {:variadic, v} ->
        v

      v ->
        v
    end)
  end

  @doc false
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

                op_applier(
                  strip_variadicity(args),
                  get_variadicity(args, opts),
                  slang_target_op: args_op
                ) >>> []

                if return_op do
                  op_applier(
                    strip_variadicity(ret),
                    get_variadicity(ret, opts),
                    slang_target_op: return_op
                  ) >>> []
                else
                  []
                end
              end
            end
          end >>> []
        end
      end
    )
  end

  # generate AST for creator for a IRDL op of symbol, like `irdl.operation`, `irdl.type`
  defp gen_creator(op, args_op, call, block, opts \\ []) do
    {name, args} = call |> Macro.decompose_call()
    args_var_ast = get_args_as_vars(args)

    input_constrains =
      args
      |> Macro.postwalk(&transform_defop_pins/1)
      |> Enum.reject(fn
        # reject hanging vars to suppress warning
        {_empty_var, _line, nil} -> true
        _ -> false
      end)

    name = Atom.to_string(name)
    creator = String.to_atom("create_" <> name)
    attr_name = String.to_atom("__slang__" <> "#{op}" <> "__")

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

  defmacro defop(call, block \\ nil) do
    gen_creator(:operation, :operands, call, block, return_op: :results, need_variadicity: true)
  end

  defp get_alias_name(def_name) do
    String.to_atom("alias_" <> Atom.to_string(def_name))
  end

  @doc """
  define an alias abbreviates lengthy types.
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

  defmacro any() do
    quote do
      use Beaver

      mlir do
        Beaver.MLIR.Dialect.IRDL.any() >>> ~t{!irdl.attribute}
      end
    end
  end

  @doc false
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

  def load(ctx, mod) when is_atom(mod) do
    apply(mod, :__slang_dialect__, [ctx])
    |> Beaver.MLIR.Transforms.canonicalize()
    |> MLIR.Pass.Composer.run!()
    |> MLIR.Operation.verify!()
    |> Beaver.MLIR.CAPI.beaverLoadIRDLDialects()
  end
end
