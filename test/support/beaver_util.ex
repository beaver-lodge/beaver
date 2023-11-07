defmodule BeaverUtils do
  @moduledoc """
  Useful utility functions to help debugging and testing
  """

  @doc """
  Convert an ast to source code and print it, and return the ast. Useful for debugging generated code
  """
  def tap_print_ast(ast) do
    ast |> tap(&IO.puts(Macro.to_string(&1)))
  end
end
