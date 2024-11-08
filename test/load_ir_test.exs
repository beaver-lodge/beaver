defmodule LoadIRTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  import Beaver.Sigils
  @example "test/fixtures/br_example.mlir"
  test "example from upstream with br", %{ctx: ctx} do
    content = File.read!(@example)

    ~m"""
    #{content}
    """.(ctx)
    |> MLIR.verify!()
  end

  test "generic form", %{ctx: ctx} do
    use Beaver
    txt = File.read!(@example)
    txt = MLIR.Module.create(txt, ctx: ctx) |> MLIR.to_string(generic: true)
    ctx = MLIR.Context.create(allow_unregistered: true, all_dialects: false)
    assert MLIR.Module.create(txt, ctx: ctx) |> MLIR.to_string() =~ "function_type = ()"
    ctx |> MLIR.Context.destroy()
  end
end
