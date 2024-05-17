defmodule Beaver do
  alias Beaver.MLIR

  @moduledoc """
  This module contains top level functions and macros for Beaver DSL for MLIR.

  Here are some of the examples of most common forms of Beaver DSL:
  - single result
  ```
  res = TOSA.add(a, b) >>> res_t
  ```
  - multiple results
  ```
  [res1, res2] = TOSA.add(a, b) >>> [res_t, res_t]
  [res1, res2] = TOSA.add(a, b) >>> res_t_list
  res_list = TOSA.add(a, b) >>> res_t_list
  ```
  - infer results
  ```
  TOSA.add(a, b) >>> :infer
  ```
  - with op
  ```
  {op, res} = TOSA.add(a, b) >>> {:op, res_t}
  ```
  - with no result
  ```
  TOSA.add(a, b) >>> []
  ```
  """

  defmacro __using__(_) do
    quote do
      import Beaver
      alias Beaver.MLIR
      import MLIR.Sigils
    end
  end

  @doc """
  This is a macro where Beaver's MLIR DSL expressions get transformed to MLIR API calls.
  This transformation will works on any expression of this form, so it is also possible to call any other function/macro rather than an Op creation function. There is one operator `>>>` for typing the result of the SSA or an argument of a block. It kind of works like the `::` in specs and types of Elixir.

  ## How it works under the hood
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
  CF.cond_br(cond0, :bb1, {:bb2, [v0]})  >>> []
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
  ```
  """
  defmacro mlir(do: dsl_block) do
    quote do
      Beaver.mlir [] do
        unquote(dsl_block)
      end
    end
  end

  defmacro mlir(opts, do: dsl_block) do
    dsl_block_ast = dsl_block |> Beaver.SSA.prewalk(&MLIR.Operation.eval_ssa/1)

    ctx_ast =
      if ctx = opts[:ctx] do
        quote do
          Kernel.var!(beaver_internal_env_ctx) = unquote(ctx)

          match?(%MLIR.Context{}, Kernel.var!(beaver_internal_env_ctx)) ||
            raise Beaver.EnvNotFoundError, MLIR.Context
        end
      end

    block_ast =
      if block = opts[:block] do
        quote do
          Kernel.var!(beaver_internal_env_block) = unquote(block)

          match?(%MLIR.Block{}, Kernel.var!(beaver_internal_env_block)) ||
            raise Beaver.EnvNotFoundError, MLIR.Block
        end
      end

    quote do
      require Beaver.Env
      alias Beaver.MLIR
      alias Beaver.MLIR.{Type, Attribute, ODS}
      import Beaver.MLIR.Sigils
      import Beaver.MLIR.Dialect.Builtin

      unquote(ctx_ast)
      unquote(block_ast)
      unquote(dsl_block_ast)
    end
  end

  # transform ast of a call into block argument bindings to variables
  defp arguments_variables(args) do
    for {{var, _type}, index} <- Enum.with_index(args) do
      quote do
        unquote(var) = Beaver.Env.block() |> Beaver.MLIR.Block.get_arg!(unquote(index))
      end
    end
  end

  # transform ast of a call into MLIR block creation
  defp add_arguments(args) do
    arg_loc_pairs =
      for {_var, type} <- args do
        {type, quote(do: MLIR.Location.from_env(__ENV__, ctx: Beaver.Env.context()))}
      end

    quote do
      tap(
        &Beaver.MLIR.Block.add_args!(
          &1,
          unquote(arg_loc_pairs),
          ctx: Beaver.Env.context()
        )
      )
    end
  end

  defmodule EnvNotFoundError do
    defexception [:message]

    @impl true
    def exception(type) when type in [MLIR.Context, MLIR.Block] do
      msg = "not a valid #{inspect(type)} in environment"
      %EnvNotFoundError{message: msg}
    end
  end

  @doc false
  def not_found(%Macro.Env{} = env) do
    {:not_found, [file: env.file, line: env.line]}
  end

  @doc false
  def parent_scope_block_caching(caller) do
    suppress_warning = quote(do: _ = Kernel.var!(beaver_internal_env_block))

    if Macro.Env.has_var?(caller, {:beaver_internal_env_block, nil}) do
      {quote do
         beaver_internal_parent_scope_block = Kernel.var!(beaver_internal_env_block)
       end,
       quote do
         Kernel.var!(beaver_internal_env_block) = beaver_internal_parent_scope_block
         unquote(suppress_warning)
       end}
    else
      {nil,
       quote do
         # erase the block in the environment to prevent unintended accessing
         Kernel.var!(beaver_internal_env_block) = Beaver.not_found(__ENV__)
         unquote(suppress_warning)
       end}
    end
  end

  defmacro block(do: body) do
    quote do
      block _() do
        unquote(body)
      end
    end
  end

  defmacro block(call, do: body) do
    {b_name, args} = Macro.decompose_call(call)
    if not is_atom(b_name), do: raise("block name must be an atom or underscore")
    {block_cache, block_restore} = parent_scope_block_caching(__CALLER__)

    quote do
      unquote(block_cache)

      Kernel.var!(beaver_internal_env_block) =
        beaver_internal_current_block =
        Beaver.Env.block(unquote({b_name, [], nil})) |> unquote(add_arguments(args))

      with region = %Beaver.MLIR.Region{} <- Beaver.Env.region() do
        Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, Beaver.Env.block())
      end

      unquote_splicing(arguments_variables(args))
      unquote(body)
      # op uses are across blocks, so op within a block can't use an API like Region.under
      unquote(block_restore)
      beaver_internal_current_block
    end
  end

  defmacro region(do: body) do
    regions =
      if Macro.Env.has_var?(__CALLER__, {:beaver_internal_env_regions, nil}) do
        quote do
          Kernel.var!(beaver_internal_env_regions)
        end
      else
        quote do
          Kernel.var!(beaver_internal_env_regions) = []
        end
      end

    quote do
      region = Beaver.MLIR.CAPI.mlirRegionCreate()
      unquote(regions)

      Beaver.MLIR.Region.under(region, fn ->
        Kernel.var!(beaver_env_region) = region
        %Beaver.MLIR.Region{} = Kernel.var!(beaver_env_region)
        unquote(body)
      end)

      Kernel.var!(beaver_internal_env_regions) =
        Kernel.var!(beaver_internal_env_regions) ++
          [region]

      Kernel.var!(beaver_internal_env_regions)
    end
  end

  def _call >>> _results do
    raise(
      "`>>>` operator is expected to be transformed away. Maybe you forget to put the expression inside the Beaver.mlir/1 macro's do block?"
    )
  end
end
