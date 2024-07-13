defmodule EnifTest do
  use Beaver
  use Beaver.Case, async: true

  @moduletag :smoke
  test "populate enif functions", test_context do
    %ENIFSupport{engine: e} = s = AddENIF.init(test_context[:ctx])
    invoker = &Beaver.ENIF.invoke(e, "add", [&1, &2])
    assert 3 == invoker.(1, 2)
    assert 1 == invoker.(-1, 2)
    ENIFSupport.destroy(s)
  end

  test "query enif functions" do
    assert :enif_binary_to_term in Beaver.ENIF.functions()
  end

  test "binary serialize", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        b = Beaver.ENIF.binary("hello") >>> :infer
        assert to_string(b) =~ "dense<[104, 101, 108, 108, 111]>"
      end
    end
    |> MLIR.Operation.verify!()
  end

  test "enif string inspected as memref", test_context do
    txt = "hello"

    ENIFStringAsMemRef.init(test_context[:ctx])
    |> tap(fn %ENIFSupport{engine: e} ->
      invoker = &Beaver.ENIF.invoke(e, "original_str", [&1])
      assert txt == invoker.(txt)
    end)
    |> tap(fn %ENIFSupport{engine: e} ->
      invoker = &Beaver.ENIF.invoke(e, "str_as_memref_get_len", [&1])
      assert String.length(txt) == invoker.(txt)
    end)
    |> tap(fn %ENIFSupport{engine: e} ->
      invoker = &Beaver.ENIF.invoke(e, "alloc_bin_and_copy", [&1])
      assert txt == invoker.(txt)

      assert_raise ErlangError, "Erlang error: \"not a binary\"", fn ->
        invoker.(1)
      end
    end)
    |> ENIFSupport.destroy()
  end
end
