defmodule MIFTest do
  use Beaver.Case, async: true

  @moduletag :smoke
  test "add two integers" do
    defmodule AddTwoInt do
      use Beaver.MIF
      alias Beaver.MIF.{Pointer, Term}

      defm add(env, a, b, error) :: Term.t() do
        ptr_a = Pointer.allocate(i64())
        ptr_b = Pointer.allocate(i64())

        arg_err =
          block do
            func.return(error)
          end

        cond_br(enif_get_int64(env, a, ptr_a) != 0) do
          cond_br(0 != enif_get_int64(env, b, ptr_b)) do
            a = Pointer.load(i64(), ptr_a)
            b = Pointer.load(i64(), ptr_b)
            sum = value llvm.add(a, b) :: i64()
            term = enif_make_int64(env, sum)
            func.return(term)
          else
            ^arg_err
          end
        else
          ^arg_err
        end
      end
    end

    {:ok, pid} = Beaver.MIF.JIT.init(AddTwoInt, name: :add_int)
    jit = Beaver.MIF.JIT.get(:add_int)
    assert String.starts_with?(AddTwoInt.__ir__(), "ML\xefR")
    assert Beaver.MIF.JIT.invoke(jit, {AddTwoInt, :add, [1, 2, :arg_err]}) == 3
    assert Beaver.MIF.JIT.invoke(jit, {AddTwoInt, :add, [1, "", :arg_err]}) == :arg_err
    assert Beaver.MIF.JIT.invoke(jit, &AddTwoInt.add/3, [1, 2, :arg_err]) == 3
    assert Beaver.MIF.JIT.invoke(jit, &AddTwoInt.add/3, [1, "", :arg_err]) == :arg_err
    :ok = Beaver.MIF.JIT.destroy(pid)
  end

  test "quick sort" do
    Beaver.MIF.JIT.init(ENIFQuickSort)
    Beaver.MIF.JIT.init([ENIFTimSort, ENIFMergeSort])
    assert ENIFQuickSort.sort(:what, :arg_err) == :arg_err
    arr = [5, 4, 3, 2, 1]
    assert ENIFQuickSort.sort(arr, :arg_err) == Enum.sort(arr)

    for i <- 0..1000 do
      arr = 0..i |> Enum.shuffle()
      assert ENIFTimSort.sort(arr, :arg_err) == Enum.sort(arr)
      assert ENIFQuickSort.sort(arr, :arg_err) == Enum.sort(arr)
      assert ENIFMergeSort.sort(arr, :arg_err) == Enum.sort(arr)
    end

    :ok = Beaver.MIF.JIT.destroy(ENIFQuickSort)
    :ok = Beaver.MIF.JIT.destroy(ENIFMergeSort)
    :ok = Beaver.MIF.JIT.destroy(ENIFTimSort)
  end
end
