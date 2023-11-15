defmodule ElixirAST do
  @moduledoc "An example to showcase the Elixir AST dialect in IRDL test."
  use Beaver.Slang, name: "ex"

  defalias any_type do
    any_of([Type.i32(), Type.f32(), Type.f64(), Type.i16(), Type.i64(), Type.string()])
  end

  defalias any_list, do: list(^any_type)

  defop atom(a = Type.string()), do: a
  defop tuple(elements = {:variadic, ^any_type}), do: elements
  defop list(elements = {:variadic, ^any_list}), do: elements

  defop binary_op(left = ^any_type, op = Type.string(), right = ^any_type) do
    [left, op, right]
  end

  defop var(name = Type.string()), do: name
  defop assign(name = Type.string(), value = ^any_type), do: [name, value]

  defop function_def(
          name = Type.string(),
          params = {:variadic, Type.string()},
          body = {:variadic, any()}
        ),
        do: [name, params, body]

  defop module_def(name = Type.string(), body = {:variadic, any()}), do: [name, body]

  defop function_call(name = Type.string(), args = {:variadic, ^any_type}) do
    [name, args]
  end

  defp gen_mlir({def, _, _} = ast) when def in [:def, :defp] do
    ast
  end

  defp gen_mlir(atom = ast) when is_atom(atom) do
    ast
  end

  # var
  defp gen_mlir({_var, [version: _, line: _], nil} = ast) do
    ast
  end

  # arg
  defp gen_mlir({_var, [generated: true, version: _], :elixir_def} = ast) do
    ast
  end

  defp gen_mlir(int = ast) when is_integer(int) do
    ast
  end

  # function call
  defp gen_mlir({func_name, _, args} = ast) when is_atom(func_name) and is_list(args) do
    ast
  end

  # erlang function call
  defp gen_mlir({{:., [line: _], [:erlang, func_name]}, _, args} = ast) when is_list(args) do
    ast
  end

  defp gen_mlir({:., [line: _], [:erlang, func_name]} = ast) when is_atom(func_name) do
    ast
  end

  # one liner do block
  defp gen_mlir({:do, {_, _, _}} = ast) do
    ast
  end

  defp gen_mlir({:do, atom} = ast) when is_atom(atom) do
    ast
  end

  defp gen_mlir([{:do, _}] = ast) do
    ast
  end

  defp gen_mlir({:__aliases__, _, _} = ast) do
    ast
  end

  # defp gen_mlir(ast) do
  #   ast |> dbg
  #   ast
  # end

  def from_ast(ast) do
    ast
    |> Macro.postwalk(&gen_mlir/1)
  end
end
