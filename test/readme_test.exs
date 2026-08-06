defmodule ReadmeTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR

  @moduletag :smoke

  @readme Path.expand("../README.md", __DIR__)

  test "README SSA DSL example", %{ctx: ctx} do
    code = block!("dsl")
    mod = compile_module!(dsl_module_source(code))
    assert %MLIR.Module{} = result = mod.build(ctx)
    assert MLIR.to_string(result) =~ "func.func @some_func"
  end

  test "README pass example", %{ctx: ctx} do
    code = block!("pass")
    pass_mod = unique_mod("ReadmePass")

    [pass_source, pipeline] = String.split(code, "\nuse Beaver\n", parts: 2)

    pass_source =
      pass_source
      |> String.replace("ToyPass", Atom.to_string(pass_mod))

    compile_module!(pass_source)

    pipeline =
      pipeline
      |> String.replace("import MLIR.Transform\n", "")
      |> String.replace("ctx = MLIR.Context.create()\n", "")
      |> String.replace("ToyPass", Atom.to_string(pass_mod))

    runner = compile_module!(pass_runner_module_source(pipeline))
    assert %MLIR.Module{} = result = runner.run(ctx)

    ir = MLIR.to_string(result)
    assert ir =~ "func.func @tosa_add"
    assert ir =~ "tosa.sub"
    refute ir =~ "tosa.add"
  end

  test "README module macro example", %{ctx: ctx} do
    code = block!("module")
    mod = compile_module!(module_macro_module_source(code))
    assert %MLIR.Module{} = result = mod.build(ctx)
    assert MLIR.to_string(result) =~ "arith.constant"
  end

  defp block!(marker) do
    pattern = ~r/<!--\s*beaver:test:#{marker}\s*-->[\s\S]*?```elixir\n(.*?)```/s

    case Regex.run(pattern, File.read!(@readme), capture: :all_but_first) do
      [code] -> code
      _ -> raise "missing beaver:test:#{marker} block in README"
    end
  end

  defp dsl_module_source(code) do
    mod = unique_mod("ReadmeDsl")

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

  defp pass_runner_module_source(pipeline) do
    mod = unique_mod("ReadmePassRunner")

    """
    defmodule #{mod} do
      use Beaver
      import MLIR.Transform

      def run(ctx) do
        #{indent(pipeline, 8)}
      end
    end
    """
  end

  defp module_macro_module_source(code) do
    mod = unique_mod("ReadmeModuleMacro")

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
      _ -> raise "failed to compile README snippet module"
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
