defmodule DocumentationExamplesTest do
  use Beaver.Case, async: true

  import ExUnit.CaptureIO

  alias Beaver.MLIR
  alias Beaver.Test.DocumentationExample

  @moduletag :smoke

  test "README quickstart runs as written, including context and module cleanup" do
    code = DocumentationExample.read!("README.md", "quickstart")

    output =
      capture_io(fn ->
        assert {:ok, _binding} = Code.eval_string(code, [], file: "README.md")
      end)

    assert output =~ "module"
    assert output =~ "arith.constant 42 : i32"
  end

  test "blocks guide SSA DSL example", %{ctx: ctx} do
    code = DocumentationExample.read!("guides/blocks.md", "control-flow")
    mod = compile_module!(dsl_module_source(code))
    assert %MLIR.Module{} = result = mod.build(ctx)

    try do
      ir = result |> MLIR.verify!() |> MLIR.to_string()
      assert ir =~ "func.func @some_func"
      assert ir =~ "cf.cond_br"
    after
      MLIR.Module.destroy(result)
    end
  end

  test "passes guide definition and pipeline run together" do
    document = "guides/passes-and-patterns.md"
    pass_mod = unique_mod("DocumentedPass")

    pass_source =
      document
      |> DocumentationExample.read!("pass-definition")
      |> String.replace("ToyPass", Atom.to_string(pass_mod))

    compile_module!(pass_source)

    pipeline =
      document
      |> DocumentationExample.read!("pass-pipeline")
      |> String.replace("ToyPass", Atom.to_string(pass_mod))

    output =
      capture_io(fn ->
        assert {:ok, _binding} = Code.eval_string(pipeline, [], file: document)
      end)

    assert output =~ "func.func @tosa_add"
    assert output =~ "tosa.sub"
    refute output =~ "tosa.add"
  end

  test "architecture guide module macro example", %{ctx: ctx} do
    code = DocumentationExample.read!("guides/architecture.md", "module")
    mod = compile_module!(module_macro_module_source(code))
    assert %MLIR.Module{} = result = mod.build(ctx)

    try do
      assert result |> MLIR.verify!() |> MLIR.to_string() =~ "arith.constant"
    after
      MLIR.Module.destroy(result)
    end
  end

  defp dsl_module_source(code) do
    mod = unique_mod("DocumentedDsl")

    """
    defmodule #{mod} do
      use Beaver
      alias Beaver.MLIR.{Attribute, Type}
      alias Beaver.MLIR.Dialect.{Func, Arith, CF}
      require Func

      def build(ctx) do
        mlir ctx: ctx do
          module do
            #{indent(code, 10)}
          end
        end
      end
    end
    """
  end

  defp module_macro_module_source(code) do
    mod = unique_mod("DocumentedModuleMacro")

    """
    defmodule #{mod} do
      use Beaver
      alias Beaver.MLIR.Dialect.Arith

      def build(ctx) do
        mlir ctx: ctx do
          #{indent(code, 10)}
        end
      end
    end
    """
  end

  defp compile_module!(source) do
    case Code.compile_string(source) do
      [{module, _bytecode} | _] -> module
      _ -> raise "failed to compile documentation snippet module"
    end
  end

  defp unique_mod(prefix) do
    Module.concat([__MODULE__, "#{prefix}#{System.unique_integer([:positive])}"])
  end

  defp indent(code, spaces) do
    code
    |> String.trim_trailing()
    |> String.split("\n")
    |> Enum.map_join("\n", &(String.duplicate(" ", spaces) <> &1))
  end
end
