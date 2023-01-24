defmodule PassTest do
  use Beaver.Case, async: true
  import ExUnit.CaptureLog
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func
  import MLIR.{Transforms}

  defmodule PassRaisingException do
    @moduledoc false
    use Beaver.MLIR.Pass, on: "func.func"

    def run(_op) do
      raise "exception in pass run"
    end
  end

  defp example_ir(test_context) do
    mlir ctx: test_context[:ctx] do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            block bb_entry() do
              v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
              Func.return(v0) >>> []
            end
          end
        end
        |> MLIR.Operation.verify!(debug: true)
      end
    end
    |> MLIR.Operation.verify!()
  end

  test "exception in run/1", test_context do
    ir = example_ir(test_context)

    assert_raise RuntimeError, ~r"Unexpected failure running passes", fn ->
      assert capture_log(fn ->
               ir
               |> MLIR.Pass.Composer.nested("func.func", [
                 PassRaisingException
               ])
               |> MLIR.Pass.Composer.run!()
             end) =~ ~r"fail to run a pass"
    end
  end

  test "pass of anonymous function", test_context do
    ir = example_ir(test_context)

    ir
    |> MLIR.Pass.Composer.append(
      {"test-pass", "builtin.module",
       fn op ->
         assert MLIR.to_string(op) =~ ~r"func.func @some_func"
         :ok
       end}
    )
    |> MLIR.Pass.Composer.run!()
  end

  test "multi level nested", test_context do
    ir = example_ir(test_context)

    assert ir
           |> canonicalize()
           |> MLIR.Pass.Composer.nested(
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
           |> MLIR.Pass.Composer.to_pipeline() =~
             ~r/func1.+func2.+func.func3\(canonicalize/
  end
end
