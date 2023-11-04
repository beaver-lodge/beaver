defmodule Beaver.Slang do
  @callback load_dialect(ctx :: Beaver.MLIR.Context.t()) :: :ok | {:error, String.t()}
  @moduledoc """
  Defining a MLIR dialect with macros in Elixir. Internally expressions are compiled to [IRDL](https://mlir.llvm.org/docs/Dialects/IRDL/)
  """
  defmacro __using__(opts) do
    name = opts |> Keyword.fetch!(:name)

    quote do
      @behaviour Beaver.Slang
      @before_compile Beaver.Slang
      import Beaver.Slang
      @__slang_dialect_name__ unquote(name)
      Module.register_attribute(__MODULE__, :__slang__operation__, accumulate: true)
      Module.register_attribute(__MODULE__, :__slang__creator__, accumulate: true)
      Module.register_attribute(__MODULE__, :__slang__type__, accumulate: true)
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      def load_dialect(ctx) do
        create_dialect(@__slang_dialect_name__, Enum.reverse(@__slang__creator__), ctx: ctx)
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
        var = {_name, _line0, nil} ->
          var

        {:=, _line0, [var, _right]} ->
          var
      end
    end
  end

  # define a IRDL op of symbol, like `irdl.operation`, `irdl.type`
  defp def_symbol_op(op, args_op, call, block, return_op \\ nil) do
    {name, args} = call |> Macro.decompose_call()

    args_var_ast = get_args_as_vars(args)
    args = args |> Macro.postwalk(&transform_defop_pins/1)
    name = Atom.to_string(name)
    creator = String.to_atom("create_" <> name)

    attr_name = String.to_atom("__slang__" <> "#{op}" <> "__")

    return_op_ast =
      if return_op do
        quote do
          import Beaver.MLIR.Dialect.IRDL, only: [{unquote(return_op), 1}]
          ret = unquote(block[:do])
          unquote(return_op)(ret) >>> []
        end
      else
        []
      end

    quote do
      Module.put_attribute(__MODULE__, unquote(attr_name), unquote(name))
      @__slang__creator__ {unquote(op), __MODULE__, unquote(creator)}
      def unquote(creator)(opts) do
        use Beaver

        Beaver.Deferred.from_opts(
          opts,
          fn ctx ->
            mlir block: opts[:block], ctx: ctx do
              import Beaver.MLIR.Dialect.IRDL, only: [{unquote(op), 1}, {unquote(args_op), 1}]

              unquote(op)(sym_name: "\"#{unquote(name)}\"") do
                region do
                  block b_op() do
                    (unquote_splicing(args))
                    unquote(args_op)(unquote_splicing(args_var_ast)) >>> []
                    unquote(return_op_ast)
                  end
                end
              end >>> []
            end
          end
        )
      end
    end
  end

  defmacro deftype(call, block \\ nil) do
    {name, args} =
      call
      |> Macro.decompose_call()

    op_ast = def_symbol_op(:type, :parameters, call, block)

    quote do
      import Beaver
      unquote(op_ast)

      def unquote(name)(unquote_splicing(get_args_as_vars(args))) do
        {:parametric, unquote(name), [unquote_splicing(get_args_as_vars(args))]}
      end
    end
  end

  defmacro defop(call, block \\ nil) do
    def_symbol_op(:operation, :operands, call, block, :results)
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

  alias Beaver.MLIR.Dialect.IRDL
  use Beaver

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

  def create_parametric(v, _opts), do: v

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
        |> MLIR.dump!()
        |> Beaver.MLIR.Transforms.canonicalize()
        |> MLIR.Pass.Composer.run!()
        |> MLIR.dump!()
      end
    )

    :ok
  end
end
