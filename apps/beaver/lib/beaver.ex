defmodule Beaver do
  @moduledoc """
  This module contains top level functions and macros for Beaver DSL for MLIR.
  """

  @doc """
  This is a macro where Beaver's MLIR DSL expressions get transformed to MLIR API calls.
  This transformation will works on any expression of this form, so it is also possible to call any other function/macro rather than an Op creation function.
  ```
  mlir do
    [res0, res1] = TestDialect.some_op(operand0, operand1, attr0: ~a{11 : i32}) :: ~t{f32}
  end
  # will be transform to:
  [res0, res1] =
    %MLIR.SSA{}
    |> MLIR.SSA.arguments([operand0, operand1, attr0: ~a{11 : i32}, attr1: ~a{11 : i32}])
    |> MLIR.SSA.results([~t{f32}])
    |> TestDialect.some_op()
  ```
  If there is no returns, add a `[]` to make the transformation effective:
  ```
  TestDialect.some_op(operand0) :: []
  ```
  By default, the creation of a terminator will be deferred in case its successor block has not been created. To force the creation of the terminator, add the `!` suffix:
  ```
  Linalg.yield!()
  ```
  To create region, call the op with a do block. The block macro works like the function definition in Elixir, and in the do block of `block` macro you can reference an argument by name. One caveat is that if it is a Op with region, it requires all arguments to be passed in one list to make it to call the macro version of the Op creation function.
  ```
  TestDialect.op_with_region [operand0, attr0: ~a{1}i32] do
    region do
      block(arg :: ~t{f32}) do
        TestDialect.some_op(arg) :: ~t{f32}
      end
    end
  end :: ~t{f32}
  """
  defmacro mlir(do: block) do
    new_block_ast = Beaver.DSL.transform_ssa(block)

    quote do
      unquote(new_block_ast)
    end
  end

  @doc false
  defmacro mlir_debug(do: block) do
    new_block_ast = Beaver.DSL.transform_ssa(block)

    env = __CALLER__
    new_block_ast |> Macro.to_string() |> IO.puts()

    block
    |> Macro.expand(env)
    # |> Macro.prewalk(&Macro.expand(&1, env))
    |> Macro.to_string()
    |> IO.puts()

    quote do
      unquote(new_block_ast)
    end
  end
end
