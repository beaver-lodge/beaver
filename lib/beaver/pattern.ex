defmodule Beaver.Pattern do
  @moduledoc """
  Beaver pattern DSL for MLIR, a PDL frontend in Elixir.
  """
  use Beaver
  alias Beaver.MLIR.Dialect.PDL
  alias Beaver.MLIR.{Attribute, Type}

  @doc false
  def insert_pat(cb, ctx, block, name, opts) do
    mlir ctx: ctx, blk: block do
      benefit = Keyword.fetch!(opts, :benefit) |> then(&Attribute.integer(Type.i16(), &1))

      PDL.pattern benefit: benefit, sym_name: Attribute.string(name) do
        region do
          block _() do
            cb.(Beaver.Env.block())
          end
        end
      end >>> []
    end
    |> MLIR.verify!()
  end

  defmacro defpat(call, do: body) do
    {name, _args} = Macro.decompose_call(call)
    block_ast = body |> Beaver.SSA.prewalk(&__MODULE__.eval_rewrite/1)

    quote do
      def unquote(name)(opts \\ [benefit: 1]) do
        &Beaver.Pattern.insert_pat(
          fn pat_block ->
            mlir ctx: &1, blk: pat_block do
              unquote(block_ast)
            end
          end,
          &1,
          &2,
          unquote(name),
          opts
        )
      end
    end
  end

  defmacro attribute() do
    quote do
      mlir do
        Beaver.MLIR.Dialect.PDL.attribute() >>> ~t{!pdl.attribute}
      end
    end
  end

  defmacro attribute(a) do
    quote do
      mlir do
        Beaver.MLIR.Dialect.PDL.attribute(value: unquote(a)) >>>
          ~t{!pdl.attribute}
      end
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
    rewrite_block_ast = block |> Beaver.SSA.prewalk(&__MODULE__.eval_rewrite/1)

    quote do
      mlir do
        Beaver.MLIR.Dialect.PDL.rewrite root: unquote(root), operand_segment_sizes: :infer do
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
          case MLIR.Value.owner(repl) do
            {:ok, owner} ->
              owner |> Beaver.MLIR.Operation.name()

            _ ->
              raise "not a pdl handler"
          end

        case pdl_handler do
          "pdl.result" ->
            Beaver.MLIR.Dialect.PDL.replace(
              opValue: unquote(root),
              replValues: repl,
              operand_segment_sizes: :infer
            ) >>> []

          "pdl.operation" ->
            Beaver.MLIR.Dialect.PDL.replace(
              opValue: unquote(root),
              replOperation: repl,
              operandSegmentSizes: :infer
            ) >>> []
        end
      end
    end
  end

  @doc false
  def gen_pdl(blk, %Beaver.Changeset{context: ctx}, %MLIR.Type{} = type) do
    mlir blk: blk, ctx: ctx do
      Beaver.MLIR.Dialect.PDL.type(constantType: type) >>> ~t{!pdl.type}
    end
  end

  def gen_pdl(blk, %Beaver.Changeset{context: ctx}, %MLIR.Attribute{} = attribute) do
    mlir blk: blk, ctx: ctx do
      Beaver.MLIR.Dialect.PDL.attribute(value: attribute) >>>
        ~t{!pdl.attribute}
    end
  end

  def gen_pdl(_, _, %MLIR.Value{} = value) do
    value
  end

  def gen_pdl(blk, %Beaver.Changeset{context: context} = changeset, f) when is_function(f, 1) do
    gen_pdl(blk, changeset, Beaver.Deferred.create(f, context))
  end

  def gen_pdl(_, _, element) do
    raise "fail to generate pdl handle for element: #{inspect(element)}"
  end

  @doc """
  The difference between a pdl.operation creation in a match body and a rewrite body:
  - in a match body, `pdl.attribute`/`pdl.operand`/`pdl.result` will be generated for unbound variables
  - in a rewrite body, all variables are considered bound before creation pdl ops
  """
  def create_operation(
        %Beaver.Changeset{
          name: op_name,
          attributes: attributes,
          operands: operands,
          location: loc,
          context: ctx,
          results: results
        } = changeset,
        block
      )
      when is_list(attributes) do
    mlir blk: block, ctx: ctx do
      results = results |> Enum.map(&gen_pdl(block, changeset, &1))

      attribute_names =
        for {k, _} <- attributes do
          k |> Atom.to_string() |> MLIR.Attribute.string()
        end

      attributes =
        for {_, a} <- attributes do
          a
        end
        |> Enum.map(&gen_pdl(block, changeset, &1))

      Beaver.MLIR.Dialect.PDL.operation(
        loc,
        operand_values: operands,
        attributeValues: attributes,
        typeValues: results,
        opName: Beaver.MLIR.Attribute.string(op_name),
        attributeValueNames: Beaver.MLIR.Attribute.array(attribute_names),
        operand_segment_sizes: :infer
      ) >>> ~t{!pdl.operation}
    end
  end

  @doc """
  Evaluate SSA as ops in a rewrite block. Note that function is only public so that it could be used in a AST.
  """
  def eval_rewrite(%Beaver.SSA{
        op: op_name,
        arguments: arguments,
        results: result_types,
        ctx: ctx,
        blk: block,
        loc: loc
      }) do
    ods_operands =
      case MLIR.ODS.Dump.lookup(op_name) do
        {:ok, op_dump} -> Enum.map(op_dump["operands"], & &1["name"])
        _ -> []
      end

    {operands, attributes} =
      Enum.split_with(arguments, fn
        {k, _v} -> Atom.to_string(k) in ods_operands
        %MLIR.Value{} -> true
      end)

    changeset = %Beaver.Changeset{
      name: op_name,
      attributes: attributes,
      operands: operands,
      location: loc,
      context: ctx
    }

    result_types_unwrap =
      case result_types do
        [:infer] -> []
        [{:op, types}] -> types |> List.wrap()
        _ -> result_types
      end
      |> Enum.map(&gen_pdl(block, changeset, &1))

    op =
      %Beaver.Changeset{changeset | results: result_types_unwrap}
      |> Beaver.Changeset.reorder_operands()
      |> create_operation(block)

    results =
      result_types_unwrap |> Enum.with_index() |> Enum.map(fn {_, i} -> result(block, op, i) end)

    results = if length(results) == 1, do: List.first(results), else: results

    case result_types do
      [{:op, _types}] ->
        {op, results}

      _ ->
        results
    end
  end

  defp result(blk, %MLIR.Value{} = v, i)
       when is_integer(i) do
    mlir blk: blk, ctx: MLIR.context(v) do
      PDL.result(v, index: Beaver.MLIR.Attribute.integer(Beaver.MLIR.Type.i32(), i)) >>>
        ~t{!pdl.value}
    end
  end

  import Beaver.MLIR.CAPI
  @doc false
  def compile_patterns(ctx, patterns, opts \\ []) do
    pattern_module = MLIR.Location.from_env(__ENV__, ctx: ctx) |> MLIR.Module.empty()
    block = Beaver.MLIR.Module.body(pattern_module)

    for p <- patterns do
      p = p.(ctx, block)

      if opts[:debug] do
        p |> MLIR.dump!()
      end
    end

    MLIR.verify!(pattern_module)

    pdl_pat_mod = mlirPDLPatternModuleFromModule(pattern_module)

    pdl_pat_mod
    |> mlirRewritePatternSetFromPDLPatternModule()
    |> mlirFreezeRewritePattern()
    |> tap(fn _ -> MLIR.Module.destroy(pattern_module) end)
  end
end
