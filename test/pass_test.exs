defmodule PassTest do
  use Beaver.Case, async: true, diagnostic: :server
  use Beaver
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func
  import MLIR.Transform

  defmodule PassRaisingException do
    @moduledoc false
    use Beaver.MLIR.Pass, on: "func.func"

    def run(_op) do
      raise "exception in pass run"
    end
  end

  defp example_ir(ctx) do
    mlir ctx: ctx do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            block do
              v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
              Func.return(v0) >>> []
            end
          end
        end
        |> MLIR.verify!()
      end
    end
    |> MLIR.verify!()
  end

  @tag capture_log: true
  test "exception in run/1", %{ctx: ctx, diagnostic_server: diagnostic_server} do
    ir = example_ir(ctx)

    assert_raise RuntimeError, ~r"Unexpected failure running passes", fn ->
      ir
      |> Beaver.Composer.nested("func.func", [
        PassRaisingException
      ])
      |> Beaver.Composer.run!()
    end

    assert Beaver.Capturer.collect(diagnostic_server) =~
             "Fail to run a pass implemented in Elixir"
  end

  test "pass of anonymous function", %{ctx: ctx} do
    ir = example_ir(ctx)

    ir
    |> Beaver.Composer.append(
      {"test-pass", "builtin.module",
       fn op ->
         assert MLIR.to_string(op) =~ ~r"func.func @some_func"
       end}
    )
    |> Beaver.Composer.run!()
  end

  test "multi level nested", %{ctx: ctx} do
    ir = example_ir(ctx)

    assert ir
           |> canonicalize()
           |> Beaver.Composer.nested(
             "func.func1",
             [
               canonicalize(),
               {:nested, "func.func2",
                [
                  canonicalize(),
                  {:nested, "func.func3",
                   [
                     canonicalize()
                   ]}
                ]}
             ]
           )
           |> Beaver.Composer.to_pipeline() =~
             ~r/func1.+func2.+func.func3\(canonicalize/
  end

  test "invalid pipeline txt", %{ctx: ctx} do
    ir = example_ir(ctx)

    assert_raise RuntimeError,
                 ~r"Unexpected failure parsing pipeline: something wrong, MLIR Textual PassPipeline Parser:1:1: error: 'something wrong' does not refer to a registered pass or pass pipeline",
                 fn ->
                   ir
                   |> Beaver.Composer.append("something wrong")
                   |> Beaver.Composer.run!()
                 end
  end
end
