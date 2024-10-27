defmodule LoadIRTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  test "example from upstream with br", %{ctx: ctx} do
    content = File.read!("test/br_example.mlir")

    ~m"""
    #{content}
    """.(ctx)
    |> MLIR.Operation.verify!()
  end
end
