defmodule Beaver do
  alias Beaver.MLIR

  @moduledoc """
  This module contains top level functions and macros for Beaver DSL for MLIR.
  """

  @doc """
  This is a macro where Beaver's MLIR DSL expressions get transformed to MLIR API calls.
  This transformation will works on any expression of this form, so it is also possible to call any other function/macro rather than an Op creation function. There is one operator `>>>` for typing the result of the SSA or an argument of a block. It kind of works like the `::` in specs and types of Elixir. Here is how is works under the hood:
  ```
  mlir do
    [res0, res1] = TestDialect.some_op(operand0, operand1, attr0: ~a{11 : i32}) >>> ~t{f32}
  end
  # will be transform to:
  [res0, res1] =
    %DSL.SSA{}
    |> DSL.SSA.arguments([operand0, operand1, attr0: ~a{11 : i32}, attr1: ~a{11 : i32}])
    |> DSL.SSA.results([~t{f32}])
    |> TestDialect.some_op()
  ```
  The SSA form will return:
  - For op with multiple result: the results of this op in a list.
  - For op with one single result: the result
  - For op with no result: the op itself (for instance, module, func, and terminators)

  If there is no returns, add a `[]` to make the transformation effective:
  ```
  TestDialect.some_op(operand0) >>> []
  ```
  To defer the creation of a terminator in case its successor block has not been created. You can pass an atom of the name in the block's call form.
  ```
  CF.cond_br(cond0, :bb1, {:bb2, [v0]})
  ```
  To create region, call the op with a do block. The block macro works like the function definition in Elixir, and in the do block of `block` macro you can reference an argument by name. One caveat is that if it is a Op with region, it requires all arguments to be passed in one list to make it to call the macro version of the Op creation function.
  ```
  TestDialect.op_with_region [operand0, attr0: ~a{1}i32] do
    region do
      block(arg >>> ~t{f32}) do
        TestDialect.some_op(arg) >>> ~t{f32}
      end
    end
  end >>> ~t{f32}
  """
  defmacro __using__(_) do
    quote do
      import Beaver
      import Beaver.Env
      alias Beaver.MLIR
      import MLIR.Sigils
    end
  end

  defmacro mlir(do: block) do
    new_block_ast = block |> Beaver.DSL.SSA.transform()

    alias Beaver.MLIR.Dialect

    alias_dialects =
      for d <-
            Dialect.Registry.dialects() do
        module_name = d |> Dialect.Registry.normalize_dialect_name()
        module_name = Module.concat([Beaver.MLIR.Dialect, module_name])

        quote do
          alias unquote(module_name)
          require unquote(module_name)
        end
      end
      |> List.flatten()

    quote do
      alias Beaver.MLIR
      alias Beaver.MLIR.Type
      alias Beaver.MLIR.Attribute
      alias Beaver.MLIR.ODS
      import Beaver.MLIR.Sigils
      unquote(alias_dialects)
      import Builtin
      import CF

      unquote(new_block_ast)
    end
  end

  defmacro block(call, do: block) do
    {
      _block_args,
      _block_opts,
      args_type_ast,
      args_var_ast,
      locations_var_ast,
      block_arg_var_ast
    } = Beaver.MLIR.DSL.Block.transform_call(call)

    {block_id, _} = Macro.decompose_call(call)
    if not is_atom(block_id), do: raise("block name must be an atom")

    block_ast =
      quote do
        unquote_splicing(args_type_ast)
        block_arg_types = [unquote_splicing(args_var_ast)]
        block_arg_locs = [unquote_splicing(locations_var_ast)]

        block = Beaver.MLIR.Block.create(block_arg_types, block_arg_locs)

        # can't put code here inside a function like Region.under, because we need to support uses across blocks
        previous_block = Beaver.MLIR.Managed.Block.get()

        Beaver.MLIR.Managed.Block.set(block)

        if region = Beaver.MLIR.Managed.Region.get() do
          # insert the block to region
          Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)
          # put the block to managed terminator => block id (name in decomposed block call)
          Beaver.MLIR.Managed.Terminator.put_block(unquote(block_id), block)
        else
          raise "no managed region found to append block"
        end

        unquote_splicing(block_arg_var_ast)
        unquote(block)
        Beaver.MLIR.Managed.Block.set(previous_block)

        block
      end

    block_ast
  end

  # TODO: check sigil_t is from MLIR
  defmacro region(do: block) do
    quote do
      region = Beaver.MLIR.CAPI.mlirRegionCreate()

      Beaver.MLIR.Region.under(region, fn ->
        unquote(block)
      end)

      [region]
    end
  end

  def _call >>> _results do
    raise(
      "`>>>` operator is expected to be transformed away. Maybe you forget to put the expression inside the Beaver.mlir/1 macro's do block?"
    )
  end

  defmacro defpat(call, do: block) do
    {name, args} = Macro.decompose_call(call)

    match = args |> Beaver.DSL.Pattern.transform_match()
    replace = block |> Beaver.DSL.Pattern.transform_rewrite()
    alias Beaver.MLIR.Dialect.PDL
    alias Beaver.MLIR.Attribute
    alias Beaver.MLIR.Type

    pdl_pattern_module_op =
      quote do
        mlir do
          module do
            PDL.pattern benefit: Attribute.integer(Type.i16(), 1),
                        sym_name: "\"#{unquote(name)}\"" do
              region do
                block some_pattern() do
                  unquote_splicing(match)
                  unquote(replace)
                end
              end
            end
          end
        end
      end

    pdl_pattern_module_op_str = pdl_pattern_module_op |> Macro.to_string()

    quote do
      def unquote(name)() do
        alias Beaver.DSL.Pattern
        pdl_pattern_module_op = unquote(pdl_pattern_module_op)

        with {:ok, pdl_pattern_module_op} <-
               Beaver.MLIR.Operation.verify(pdl_pattern_module_op, dump_if_fail: true) do
        else
          :fail ->
            code_gen_by_macro = unquote(pdl_pattern_module_op_str)

            IO.puts("""
            Here is the code generated by the #{unquote(name)} macro:
            #{code_gen_by_macro}
            """)

            raise "fail to verify generated pattern"
        end

        pdl_pattern_module_op
      end
    end
  end

  def concrete(%MLIR.CAPI.MlirOperation{} = op) do
    MLIR.Operation.to_prototype(op)
  end

  def container(module = %MLIR.CAPI.MlirModule{}) do
    MLIR.Operation.from_module(module)
  end

  def container(%{
        operands: %Beaver.Walker{container: container},
        attributes: %Beaver.Walker{container: container},
        results: %Beaver.Walker{container: container},
        successors: %Beaver.Walker{container: container},
        regions: %Beaver.Walker{container: container}
      }) do
    container
  end

  def container(container) do
    container
  end
end
