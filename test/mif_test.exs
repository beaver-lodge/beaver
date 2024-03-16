defmodule MIFTest do
  use Beaver.Case, async: true

  @moduletag :smoke
  test "add two integers", test_context do
    defmodule AddTwoInt do
      use Beaver.MIF
      alias Beaver.MIF.{BEAM, Pointer}

      defm add(a, b, error) do
        env = BEAM.env()
        ptr_a = Pointer.allocate(i64())
        ptr_b = Pointer.allocate(i64())

        arg_err =
          block do
            op func.return(error) :: []
          end

        cond_br(enif_get_int64(env, a, ptr_a) != 0) do
          cond_br(0 != enif_get_int64(env, b, ptr_b)) do
            a = Pointer.load(i64(), ptr_a)
            b = Pointer.load(i64(), ptr_b)
            add_op = op llvm.add(a, b) :: i64()
            sum = result_at(add_op, 0)
            term = enif_make_int64(env, sum)
            op func.return(term) :: []
          else
            ^arg_err
          end
        else
          ^arg_err
        end
      end
    end

    Beaver.MIF.init_jit(AddTwoInt)
    assert AddTwoInt.add(1, 2, :arg_err) == 3
    assert AddTwoInt.add(1, "", :arg_err) == :arg_err
    Beaver.MIF.destroy_jit(AddTwoInt)
  end
end
