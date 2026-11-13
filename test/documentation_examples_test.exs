defmodule DocumentationExamplesTest do
  use Beaver.Case, async: true
  import ExUnit.CaptureIO
  alias Beaver.Test.DocumentationExample
  @moduletag :smoke
  test "README quickstart runs as written, including context and module cleanup" do
    output = DocumentationExample.read!("README.md", "quickstart") |> run!("README.md")
    assert output =~ "module"
    assert output =~ "arith.constant 42 : i32"
  end

  test "blocks guide SSA DSL example", %{ctx: ctx} do
    result = Beaver.Dummy.documented_control_flow(ctx)

    MLIR.Module.with_module(result, fn _ ->
      ir = result |> MLIR.verify!() |> MLIR.to_string()
      assert ir =~ "func.func @some_func"
      assert ir =~ "cf.cond_br"
    end)
  end

  test "passes guide definition and pipeline run together" do
    document = "guides/passes-and-patterns.md"

    output =
      ["pass-definition", "pass-pipeline"]
      |> Enum.map_join("\n", &DocumentationExample.read!(document, &1))
      |> String.replace("ToyPass", "DocumentedPass#{System.unique_integer([:positive])}")
      |> run!(document)

    assert output =~ "func.func @tosa_add"
    assert output =~ "tosa.sub"
    refute output =~ "tosa.add"
  end

  defp run!(code, document) do
    capture_io(fn -> assert {:ok, _binding} = Code.eval_string(code, [], file: document) end)
  end
end
