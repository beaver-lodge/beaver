defmodule LoadIRTest do
  use ExUnit.Case
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  test "example from upstream with br" do
    content = File.read!("test/br_example.mlir")

    ~m"""
    #{content}
    """
  end
end
