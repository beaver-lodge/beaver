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
    txt = File.read!(@example) |> MLIR.Module.create(ctx: ctx) |> MLIR.to_string(generic: true)
    generic_snippet = "function_type = ()"
    assert txt =~ generic_snippet
    ctx_generic = MLIR.Context.create(allow_unregistered: true, all_dialects: false)
    assert MLIR.Module.create(txt, ctx: ctx_generic) |> MLIR.to_string() =~ generic_snippet
    ctx_generic |> MLIR.Context.destroy()
  end
end
