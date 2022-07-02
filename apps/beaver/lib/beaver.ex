defmodule Beaver do
  @moduledoc """
  This module contains top level functions and macros for Beaver DSL for MLIR.
  """

  @doc """
  This is a macro where Beaver's MLIR DSL expressions get transformed to MLIR API calls.
  This transformation will works on any expression of this form, so it is also possible to call any other function/macro rather than an Op creation function.
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
  If there is no returns, add a `[]` to make the transformation effective:
  ```
  TestDialect.some_op(operand0) >>> []
  ```
  By default, the creation of a terminator will be deferred in case its successor block has not been created. To force the creation of the terminator, add the `!` suffix:
  ```
  Linalg.yield!()
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
      import Beaver.MLIR
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

  def _call >>> _results do
    raise(
      "`>>>` operator is expected to be transformed away. Maybe you forget to put the expression inside the Beaver.mlir/1 macro's do block?"
    )
  end
end
