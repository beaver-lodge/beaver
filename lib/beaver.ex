defmodule Beaver do
  alias Beaver.MLIR
  require Beaver.Env

  @moduledoc """
  This module contains top level functions and macros for Beaver DSL for MLIR.

  `use Beaver` will import and alias the essential modules and functions for Beaver'DSL for MLIR. It also imports the sigils used in the DSL.

  ## Basic usage
  To create an MLIR operation and insert it to a block, you can use the `mlir` macro. Here is an example of creating a constant operation:
      iex> use Beaver
      iex> {ctx, blk} = {MLIR.Context.create(), MLIR.Block.create()}
      iex> alias Beaver.MLIR.Dialect.Arith
      iex> const = mlir ctx: ctx, ip: blk do
      iex>   Arith.constant(value: ~a{64: i32}) >>> ~t{i32}
      iex> end
      iex> const |> MLIR.Value.owner!() |> MLIR.verify!()
      iex> MLIR.Block.destroy(blk)
      iex> MLIR.Context.destroy(ctx)

  ## Naming conventions
  There are some concepts in Elixir and MLIR shared the same name, so it is encouraged to use the following naming conventions for variables:
  - use `ctx` for MLIR context
  - use `blk` for MLIR block

  ## Lazy creation of MLIR entities
  In Beaver, various functions and sigils might return a function with a signature like `MLIR.Context.t() -> MLIR.Type.t()`, if the context is not provided. When a creator gets passed to a SSA expression, it will be called with the context to create the entity or later the operation. Deferring the creation of the entity until context available is intended to keep the DSL code clean and succinct. For more information, see the docs of module `Beaver.Deferred`.
  """

  defmacro __using__(_) do
    quote do
      import Beaver
      alias Beaver.MLIR
      import Beaver.Sigils
    end
  end

  @doc """
  Transform the given DSL block but without specifying the context and block.

  This is useful when you want to generate partials of quoted code and doesn't want to pass around the context and block.
  """
  defmacro mlir(do: dsl_block) do
    quote do
      Beaver.mlir [] do
        unquote(dsl_block)
      end
    end
  end

  @doc """
  Macro where Beaver's MLIR DSL expressions get transformed to MLIR API calls.


  This transformation will works on any expression of the `>>>` form, so it is also possible to call any other vanilla Elixir function/macro.

  ## How it works under the hood
  ```
  [res0, res1] = TestDialect.some_op(operand0, operand1, attr0: ~a{11 : i32}) >>> ~t{f32}
  ```
  will be transform to:
  ```
  [res0, res1] =
    %SSA{}
    |> SSA.arguments([operand0, operand1, attr0: ~a{11 : i32}, attr1: ~a{11 : i32}])
    |> SSA.results([~t{f32}])
    |> TestDialect.some_op()
  ```
  and `TestDialect.some_op()` is an Elixir function to actually create the MLIR operation. So it is possible to replace the left hand of the `>>>` operator with any Elixir function.
  """
  defmacro mlir(opts, do: dsl_block) do
    dsl_block_ast = dsl_block |> Beaver.SSA.prewalk(&MLIR.Operation.eval_ssa/1)

    ctx_ast =
      if ctx = Beaver.Deferred.fetch_context(opts) do
        quote do
          Kernel.var!(beaver_internal_env_ctx) = unquote(ctx)

          match?(%MLIR.Context{}, Kernel.var!(beaver_internal_env_ctx)) ||
            raise CompileError,
                  Beaver.Env.compile_err_msg(MLIR.Context, unquote(Macro.escape(__CALLER__)))
        end
      end

    ip_ast =
      if ip = Beaver.Deferred.fetch_insertion_point(opts) do
        quote do
          Kernel.var!(beaver_internal_env_ip) = unquote(ip)

          Beaver.Env.valid_insertion_point?(Kernel.var!(beaver_internal_env_ip)) ||
            raise CompileError,
                  Beaver.Env.compile_err_msg("insertion point", unquote(Macro.escape(__CALLER__)))
        end
      end

    quote do
      require Beaver.Env
      alias Beaver.MLIR
      alias Beaver.MLIR.{Type, Attribute, ODS}
      import Beaver.Sigils
      import Beaver.MLIR.Dialect.Builtin

      unquote(ctx_ast)
      unquote(ip_ast)
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

  @doc false
  def not_found(%Macro.Env{} = env) do
    {:not_found, [file: env.file, line: env.line]}
  end

  @doc false
  def parent_scope_ip_caching(caller) do
    suppress_warning = quote(do: _ = Kernel.var!(beaver_internal_env_ip))

    if Macro.Env.has_var?(caller, {:beaver_internal_env_ip, nil}) do
      {quote do
         beaver_internal_parent_scope_ip = Kernel.var!(beaver_internal_env_ip)
       end,
       quote do
         Kernel.var!(beaver_internal_env_ip) = beaver_internal_parent_scope_ip
         unquote(suppress_warning)
       end}
    else
      {nil,
       quote do
         # erase the block in the environment to prevent unintended accessing
         Kernel.var!(beaver_internal_env_ip) = Beaver.not_found(__ENV__)
         unquote(suppress_warning)
       end}
    end
  end

  @doc """
  Create an anonymous block.

  The block can be used to call CAPI.
  """
  defmacro block(do: body) do
    quote do
      block _() do
        unquote(body)
      end
    end
  end

  @doc """
  Create a named block.

  The block macro works like the function definition in Elixir, and in the do block of `block` macro you can reference an argument by name.
  It should follow Elixir's lexical scoping rules and can be referenced by `Beaver.Env.block/1`

  > #### MLIR doesn't have named block {: .info}
  >
  > Note that the idea of named block is Beaver's concept. MLIR doesn't have it.
  """
  defmacro block(call, do: body) do
    {b_name, args} = Macro.decompose_call(call)
    if not is_atom(b_name), do: raise("block name must be an atom or underscore")
    {ip_cache, ip_restore} = parent_scope_ip_caching(__CALLER__)

    quote do
      unquote(ip_cache)

      Kernel.var!(beaver_internal_env_ip) =
        beaver_internal_current_ip =
        Beaver.Env.block(unquote({b_name, [], nil})) |> unquote(add_arguments(args))

      with region = %Beaver.MLIR.Region{} <- Beaver.Env.region() do
        Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, Beaver.Env.block())
      end

      unquote_splicing(arguments_variables(args))
      unquote(body)
      # op uses are across blocks, so op within a block can't use an API like Region.under
      unquote(ip_restore)
      beaver_internal_current_ip
    end
  end

  @doc """
  Create MLIR region. Calling the macro within a do block will create an operation with the region.

  One caveat is that if it is a Op with region, it requires all arguments to be passed in one list to make it to call the macro version of the Op creation function.
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

  @doc """
  Create an MLIR SSA expression with the given arguments and results.

  The right hand of operator `>>>` is used to typing the result of the SSA or an argument of a block. It kind of works like the `::` in specs and types of Elixir.

  ## The return of SSA expression
  By default SSA expression will return the MLIR values or the operation created.
  - For op with multiple result: the results of this op in a list.
  - For op with one single result: the result
  - For op with no result: the op itself (for instance, module, func, and terminators)

  ## Different result declarations
  There are several ways to declare the result of an SSA expression. The most common cases are single result and multi-results.

  - Single result
  ```
  res = Foo.bar(a, b) >>> res_t
  ```
  - Multiple results
  ```
  [res1, res2] = Foo.bar(a, b) >>> [res_t, res_t]
  [res1, res2] = Foo.bar(a, b) >>> res_t_list
  res_list = Foo.bar(a, b) >>> res_t_list
  ```
  - Zero result
  ```
  Foo.bar(a, b) >>> []
  ```

  - Enable type inference,
  ```
  Foo.bar(a, b) >>> :infer
  ```
  - To return the op together with the result
  ```
  {op, res} = Foo.bar(a, b) >>> {:op, types}
  ```

  """
  def _call >>> _results do
    raise(
      "`>>>` operator is expected to be transformed away. Maybe you forget to put the expression inside the Beaver.mlir/1 macro's do block?"
    )
  end
end
