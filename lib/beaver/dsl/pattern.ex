defmodule Beaver.DSL.Pattern.Env do
  defstruct ctx: nil, block: nil, loc: nil
end

defmodule Beaver.DSL.Pattern do
  @doc """
  PDL frontend
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.PDL
  alias Beaver.MLIR.{Attribute, Type}
  import MLIR.Sigils
  import Beaver
  require Beaver.MLIR
  require Beaver.MLIR.CAPI
  alias Beaver.DSL.Pattern.Env

  defmacro defpat(call, do: block) do
    {name, _args} = Macro.decompose_call(call)
    block_ast = block |> Beaver.DSL.SSA.prewalk(&__MODULE__.eval_rewrite/2)

    pdl_pattern_module_op =
      quote do
        mlir ctx: ctx do
          module do
            benefit = Keyword.get(opts, :benefit, 1)

            PDL.pattern benefit: Attribute.integer(Type.i16(), benefit),
                        sym_name: "\"#{unquote(name)}\"" do
              region do
                block _pattern_block() do
                  unquote(block_ast)
                end
              end
            end >>> []
          end
        end
      end

    pdl_pattern_module_op_ast_dump = pdl_pattern_module_op |> Macro.to_string()

    quote do
      def unquote(name)(opts \\ [benefit: 1]) do
        Beaver.Deferred.from_opts(
          opts,
          fn ctx ->
            alias Beaver.DSL.Pattern
            pdl_pattern_module_op = unquote(pdl_pattern_module_op)

            with {:ok, op} <-
                   Beaver.MLIR.Operation.verify(pdl_pattern_module_op, dump_if_fail: true) do
              op
            else
              :fail ->
                require Logger

                Logger.info("""
                Here is the code generated by the #{unquote(name)}:
                #{unquote(pdl_pattern_module_op_ast_dump)}
                """)

                raise "fail to verify generated pattern"
            end
          end
        )
      end
    end
  end

  defmacro attribute(a) do
    quote do
      Beaver.MLIR.Dialect.PDL.attribute(value: unquote(a)) >>>
        ~t{!pdl.attribute}
    end
  end

  defmacro type() do
    quote do
      mlir do
        Beaver.MLIR.Dialect.PDL.type() >>> ~t{!pdl.type}
      end
    end
  end

  defmacro type(t) do
    quote do
      mlir do
        Beaver.MLIR.Dialect.PDL.type(constantType: unquote(t)) >>> ~t{!pdl.type}
      end
    end
  end

  defmacro value() do
    quote do
      mlir do
        Beaver.MLIR.Dialect.PDL.operand() >>> ~t{!pdl.value}
      end
    end
  end

  defmacro rewrite(root, do: block) do
    rewrite_block_ast = block |> Beaver.DSL.SSA.prewalk(&__MODULE__.eval_rewrite/2)

    quote do
      mlir do
        Beaver.MLIR.Dialect.PDL.rewrite [
          unquote(root),
          operand_segment_sizes: Beaver.MLIR.ODS.operand_segment_sizes([1, 0])
        ] do
          region do
            block _rewrite_block() do
              unquote(rewrite_block_ast)
            end
          end
        end >>> []
      end
    end
  end

  defmacro replace(root, opts) do
    quote do
      mlir do
        opts = unquote(opts)
        repl = opts |> Keyword.fetch!(:with)

        pdl_handler =
          with {:ok, owner} <- MLIR.Value.owner(repl) do
            owner |> Beaver.MLIR.Operation.name()
          else
            _ ->
              raise "not a pdl handler"
          end

        case pdl_handler do
          "pdl.result" ->
            Beaver.MLIR.Dialect.PDL.replace([
              unquote(root),
              repl,
              operand_segment_sizes: Beaver.MLIR.ODS.operand_segment_sizes([1, 0, 1])
            ]) >>> []

          "pdl.operation" ->
            Beaver.MLIR.Dialect.PDL.replace([
              unquote(root),
              repl,
              operand_segment_sizes: Beaver.MLIR.ODS.operand_segment_sizes([1, 1, 0])
            ]) >>> []
        end
      end
    end
  end

  @doc false
  def gen_pdl(%Env{} = env, %MLIR.CAPI.MlirType{} = type) do
    mlir block: env.block, ctx: env.ctx do
      Beaver.MLIR.Dialect.PDL.type(constantType: type) >>> ~t{!pdl.type}
    end
  end

  def gen_pdl(%Env{} = env, %Beaver.MLIR.CAPI.MlirAttribute{} = attribute) do
    mlir block: env.block, ctx: env.ctx do
      Beaver.MLIR.Dialect.PDL.attribute(value: attribute) >>>
        ~t{!pdl.attribute}
    end
  end

  def gen_pdl(_env, %MLIR.Value{} = value) do
    value
  end

  def gen_pdl(%Env{ctx: ctx} = env, f) when is_function(f, 1) do
    gen_pdl(env, f.(ctx))
  end

  def gen_pdl(env, element) do
    dbg(env)
    dbg(element, structs: false)
    raise "not supported"
  end

  @doc """
  The difference between a pdl.operation creation in a match body and a rewrite body:
  - in a match body, `pdl.attribute`/`pdl.operand`/`pdl.result` will be generated for unbound variables
  - in a rewrite body, all variables are considered bound before creation pdl ops
  """
  def create_operation(
        %Env{ctx: ctx, block: block, loc: loc} = env,
        op_name,
        operands,
        attributes,
        results
      )
      when is_list(attributes) do
    mlir block: block, ctx: ctx do
      results = results |> Enum.map(&gen_pdl(env, &1))

      attribute_names =
        for {k, _} <- attributes do
          k |> Atom.to_string() |> MLIR.Attribute.string()
        end

      attributes =
        for {_, a} <- attributes do
          a
        end
        |> Enum.map(&gen_pdl(env, &1))

      Beaver.MLIR.Dialect.PDL.operation(
        loc,
        operands,
        attributes,
        results,
        opName: Beaver.MLIR.Attribute.string(op_name),
        attributeValueNames: Beaver.MLIR.Attribute.array(attribute_names),
        operand_segment_sizes:
          Beaver.MLIR.ODS.operand_segment_sizes([
            length(operands),
            length(attributes),
            length(results)
          ])
      ) >>> ~t{!pdl.operation}
    end
  end

  @doc """
  Evaluate SSA as ops in a rewrite block. Note that function is only public so that it could be used in a AST.
  """
  # TODO: change it to result types
  def eval_rewrite(
        op_name,
        %Beaver.DSL.SSA{
          arguments: arguments,
          results: result_types,
          ctx: ctx,
          block: block,
          loc: loc
        }
      ) do
    attributes = for {_k, _a} = a <- arguments, do: a
    operands = for %MLIR.Value{} = o <- arguments, do: o
    env = %Env{ctx: ctx, block: block, loc: loc}

    result_types_unwrap =
      case result_types do
        [:infer] -> []
        [{:op, types}] -> types |> List.wrap()
        _ -> result_types
      end
      |> Enum.map(&gen_pdl(env, &1))

    op =
      create_operation(
        env,
        op_name,
        operands,
        attributes,
        result_types_unwrap
      )

    op = %{op | safe_to_print: false}

    results =
      result_types_unwrap |> Enum.with_index() |> Enum.map(fn {_, i} -> result(env, op, i) end)

    results = if length(results) == 1, do: List.first(results), else: results

    case result_types do
      [{:op, _types}] ->
        {op, results}

      _ ->
        results
    end
  end

  defp result(%Env{block: block, ctx: ctx}, %Beaver.MLIR.Value{} = v, i)
       when is_integer(i) do
    mlir block: block, ctx: ctx do
      v = %{v | safe_to_print: false}

      r =
        PDL.result(v, index: Beaver.MLIR.Attribute.integer(Beaver.MLIR.Type.i32(), i)) >>>
          ~t{!pdl.value}

      %{r | safe_to_print: false}
    end
  end
end
