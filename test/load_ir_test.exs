defmodule LoadIRTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  test "example from upstream with br", test_context do
    content = File.read!("test/br_example.mlir")

    ~m"""
    #{content}
    """.(test_context[:ctx])
    |> MLIR.Operation.verify!()
  end
end
