defmodule PassTest do
  use Beaver.Case, async: true, diagnostic: :server
  use Beaver
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func
  import MLIR.Transform

  defmodule PassRaisingException do
    @moduledoc false
    use Beaver.MLIR.Pass, on: "func.func"

    def run(_op, _state) do
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

  test "exception in run/1", %{ctx: ctx} do
    ir = example_ir(ctx)

    assert_raise ArgumentError,
                 ~r"PassTest.PassRaisingException.run\/2.+Fail to run a pass implemented in Elixir"s,
                 fn ->
                   ir
                   |> Beaver.Composer.nested("func.func", [
                     PassRaisingException
                   ])
                   |> Beaver.Composer.run!()
                 end
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

  test "parallel processing func.func", %{ctx: ctx} do
    Beaver.Dummy.gigantic(ctx)
    |> Beaver.Composer.nested("func.func", {"DoNothingHere", "func.func", fn _ -> :ok end})
    |> Beaver.Composer.run!()
  end

  defmodule AllCallbacks do
    @moduledoc false
    use Beaver.MLIR.Pass, on: "func.func"

    @counter %{
      destruct: 0,
      initialize: 0,
      clone: 0,
      run: 0
    }
    def initialize(_ctx, nil) do
      {:ok, pid} =
        Agent.start_link(
          fn ->
            %{@counter | initialize: 1}
          end,
          name: __MODULE__
        )

      {:ok, {:init, pid}}
    end

    def initialize(_ctx, {:init, pid}) do
      Agent.update(
        pid,
        fn state ->
          update_in(state.initialize, &(&1 + 1))
        end
      )

      {:ok, {:re_init, pid}}
    end

    def clone({:init, pid}) do
      Agent.update(pid, fn state -> update_in(state.clone, &(&1 + 1)) end)
      {:clone, pid}
    end

    def run(op, {:clone, pid} = state) do
      assert MLIR.Operation.name(op) == "func.func"
      Agent.update(pid, fn state -> update_in(state.run, &(&1 + 1)) end)
      state
    end

    def destruct({:clone, pid}) do
      Agent.update(pid, fn state -> update_in(state.destruct, &(&1 + 1)) end)
    end

    def destruct({:init, pid}) do
      Agent.update(pid, fn state -> update_in(state.destruct, &(&1 + 1)) end)
    end

    def destruct({:re_init, pid}) do
      Agent.update(pid, fn state -> update_in(state.destruct, &(&1 + 1)) end)
    end
  end

  test "full life cycle", %{ctx: ctx} do
    n = 100

    Beaver.Dummy.gigantic(ctx, n)
    |> Beaver.Composer.nested("func.func", AllCallbacks)
    |> canonicalize()
    |> Beaver.Composer.run!()

    assert %{run: ^n, clone: clone, initialize: 1, destruct: destruct} =
             Agent.get(AllCallbacks, & &1, :infinity)

    assert clone + 1 == destruct
    assert :ok = Agent.stop(AllCallbacks)
  end

  test "run pm multiple times", %{ctx: ctx} do
    n = 100

    op = Beaver.Dummy.gigantic(ctx, n)

    composer =
      Beaver.Composer.new(ctx: ctx)
      |> Beaver.Composer.nested("func.func", AllCallbacks)
      |> canonicalize()

    pm = Beaver.Composer.init(composer)

    assert :ok = MLIR.PassManager.run(pm, op)

    assert %{run: ^n, clone: clone0, initialize: 1, destruct: 0} =
             Agent.get(AllCallbacks, & &1, :infinity)

    assert :ok = MLIR.PassManager.run(pm, op)
    assert :ok = MLIR.PassManager.destroy(pm)

    n2 = n * 2

    assert %{run: ^n2, clone: clone, initialize: 2, destruct: destruct} =
             Agent.get(AllCallbacks, & &1, :infinity)

    assert clone0 == clone
    assert clone + 1 == destruct
    assert :ok = Agent.stop(AllCallbacks)
  end

  defmodule ErrInit do
    @moduledoc false
    use Beaver.MLIR.Pass, on: "func.func"

    def initialize(_ctx, nil) do
      {:error, "new state used in cleanup"}
    end

    def destruct(state) do
      {:ok, _} = Agent.start_link(fn -> state end, name: __MODULE__)
      :ok
    end
  end

  test "run init with err", %{ctx: ctx} do
    op = Beaver.Dummy.gigantic(ctx, 10)

    composer =
      Beaver.Composer.new(ctx: ctx)
      |> Beaver.Composer.nested("func.func", ErrInit)
      |> canonicalize()

    pm = Beaver.Composer.init(composer)
    {:error, diagnostics} = MLIR.PassManager.run(pm, op)

    assert MLIR.Diagnostic.format(diagnostics) =~
             "Fail to initialize a pass implemented in Elixir"

    assert :ok = MLIR.PassManager.destroy(pm)
    assert "new state used in cleanup" = Agent.get(ErrInit, fn s -> s end)
    assert :ok = Agent.stop(ErrInit)
  end
end
