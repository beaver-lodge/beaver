defmodule POCTest do
  alias Beaver.MIF.Expander, as: POC
  alias Beaver.MLIR
  use ExUnit.Case, async: true

  defp env(string) do
    quoted = Code.string_to_quoted!(string, columns: true)
    {_, _, env} = POC.expand(quoted, "example.exs")
    env
  end

  defp state(string) do
    quoted = Code.string_to_quoted!(string, columns: true)
    {_, state, _} = POC.expand(quoted, "example.exs")
    state
  end

  defp compile(quoted) do
    {_, state, _} = POC.expand(quoted, "example.exs")
    state.mlir.mod
  end

  defp vars(string), do: Enum.sort(state(string).vars)
  defp locals(string), do: Enum.sort(state(string).locals)
  defp remotes(string), do: Enum.sort(state(string).remotes)

  test "locals" do
    assert locals("foo()") == [foo: 0]
    assert locals("foo(1, 2)") == [foo: 2]
    assert locals("foo(1, bar(2))") == [bar: 1, foo: 2]
  end

  # This test shows we can track locals inside containers,
  # as an example of traversal.
  test "containers" do
    assert locals("[foo()]") == [foo: 0]
    assert locals("[foo() | bar(1, 2)]") == [bar: 2, foo: 0]
    assert locals("{foo(), bar(1, 2)}") == [bar: 2, foo: 0]
    assert locals("{foo(), bar(1, 2), baz(3)}") == [bar: 2, baz: 1, foo: 0]
    assert locals("%{foo() => bar(1, 2)}") == [bar: 2, foo: 0]
  end

  # This test shows we can track locals inside unquotes.
  test "quote" do
    assert locals("quote do: foo()") == []
    assert locals("quote do: unquote(foo())") == [foo: 0]
    assert locals("quote line: line(), do: foo()") == [line: 0]
  end

  test "vars" do
    assert vars("var = 123") == [var: nil]
    assert vars("^var = 123") == []
  end

  test "remotes" do
    assert remotes(":lists.flatten([])") == [{:lists, :flatten, 1}]
    assert remotes("List.flatten([])") == [{List, :flatten, 1}]
  end

  describe "defmodule" do
    test "requires module" do
      env = env("defmodule Foo, do: :ok")
      # assert Macro.Env.required?(env, Foo)
      assert env.context_modules == [Foo]
      refute env.module

      env = env("defmodule Foo.Bar, do: :ok")
      # assert Macro.Env.required?(env, Foo.Bar)
      assert env.context_modules == [Foo.Bar]
      refute env.module
    end

    test "alias module" do
      assert {Foo.Bar, :flatten, 1} in remotes(
               "defmodule Foo do defmodule Bar do Bar.flatten([]) end end"
             )

      assert {Bar, :flatten, 1} in remotes(
               "defmodule Foo do defmodule Elixir.Bar do Bar.flatten([]) end end"
             )
    end
  end

  describe "alias/2" do
    test "defines aliases" do
      assert remotes("alias List, as: L; L.flatten([])") == [{List, :flatten, 1}]
    end
  end

  describe "require/2" do
    defmacro discard_require(_discard), do: :ok

    test "requires modules" do
      # The macro discards the content, so if the module is required,
      # the macro is invoked and contents are discarded
      assert locals("POCTest.discard_require(foo())") == [foo: 0]
      assert locals("require POCTest; POCTest.discard_require(foo())") == []
    end
  end

  describe "import/2" do
    defmacro discard_import(_discard), do: :ok

    test "imports modules" do
      # The macro discards the content, so if the module is imported,
      # the macro is invoked and contents are discarded
      assert locals("discard_import(foo())") == [discard_import: 1, foo: 0]
      assert locals("import POCTest; discard_import(foo())") == []
    end
  end

  describe "def" do
    test "return original arg" do
      quote do
        defmodule ReturnPassedArg do
          import Beaver.MIF.Prelude
          alias Beaver.MIF.Pointer
          alias Beaver.MIF.Term
          alias Beaver.MIF.Env
          def foo(a :: Beaver.MIF.Term.t()) :: Beaver.MIF.Term.t(), do: func.return(a)

          defm bar(env, a) do
            alias Beaver.MIF.Term
            b = foo(a) :: Term.t()
            func.return(b)
          end
        end
      end
      |> compile()
      |> MLIR.dump!()
      |> MLIR.Operation.verify!()
      |> tap(fn m ->
        {:ok, pid} = Beaver.MIF.JIT.init(m, name: :return_this)
        jit = Beaver.MIF.JIT.get(:return_this)
        assert Beaver.MIF.JIT.invoke(jit, {ReturnPassedArg, :bar, [:identical]}) == :identical
        :ok = Beaver.MIF.JIT.destroy(pid)
      end)
    end
  end
end
