defmodule ENIFTimSort do
  use Beaver.MIF
  require Beaver.Env
  alias Beaver.MIF.{Pointer, Term, Env}

  defm insertion_sort(arr :: Pointer.t(), left :: i32(), right :: i32()) do
    start_i = left + 1
    start = Pointer.element_ptr(Term.t(), arr, start_i)
    n = right - start_i + 1

    for_loop {temp, i} <- {Term.t(), start, n} do
      i = op index.casts(i) :: i32()
      i = result_at(i, 0) + start_i
      j_ptr = Pointer.allocate(i32())
      Pointer.store(i - 1, j_ptr)

      while_loop(
        Pointer.load(i32(), j_ptr) >= left &&
          Pointer.load(Term.t(), Pointer.element_ptr(Term.t(), arr, Pointer.load(i32(), j_ptr))) >
            temp
      ) do
        j = Pointer.load(i32(), j_ptr)

        Pointer.store(
          Pointer.load(Term.t(), Pointer.element_ptr(Term.t(), arr, j)),
          Pointer.element_ptr(Term.t(), arr, j + 1)
        )

        Pointer.store(j - 1, j_ptr)
      end

      j = Pointer.load(i32(), j_ptr)
      Pointer.store(temp, Pointer.element_ptr(Term.t(), arr, j + 1))
    end

    op func.return() :: []
  end

  defm merge(arr :: Pointer.t(), l :: i32(), m :: i32(), r :: i32()) do
    n1 = m - l + 1
    n2 = r - m

    left_temp = Pointer.allocate(Term.t(), n1)
    right_temp = Pointer.allocate(Term.t(), n2)

    for_loop {element, i} <- {Term.t(), Pointer.element_ptr(Term.t(), arr, l), n1} do
      i = op index.casts(i) :: i32()
      i = result_at(i, 0)
      Pointer.store(element, Pointer.element_ptr(Term.t(), left_temp, i))
    end

    for_loop {element, j} <- {Term.t(), Pointer.element_ptr(Term.t(), arr, m + 1), n2} do
      j = op index.casts(j) :: i32()
      j = result_at(j, 0)
      Pointer.store(element, Pointer.element_ptr(Term.t(), right_temp, j))
    end

    i_ptr = Pointer.allocate(i32())
    j_ptr = Pointer.allocate(i32())
    k_ptr = Pointer.allocate(i32())

    zero_const = op arith.constant(value: Attribute.integer(i32(), 0)) :: i32()
    zero = result_at(zero_const, 0)
    Pointer.store(zero, i_ptr)
    Pointer.store(zero, j_ptr)
    Pointer.store(l, k_ptr)

    while_loop(Pointer.load(i32(), i_ptr) < n1 && Pointer.load(i32(), j_ptr) < n2) do
      i = Pointer.load(i32(), i_ptr)
      j = Pointer.load(i32(), j_ptr)
      k = Pointer.load(i32(), k_ptr)

      left_term = Pointer.load(Term.t(), Pointer.element_ptr(Term.t(), left_temp, i))
      right_term = Pointer.load(Term.t(), Pointer.element_ptr(Term.t(), right_temp, j))

      struct_if(enif_compare(left_term, right_term) <= 0) do
        Pointer.store(
          Pointer.load(Term.t(), Pointer.element_ptr(Term.t(), left_temp, i)),
          Pointer.element_ptr(Term.t(), arr, k)
        )

        Pointer.store(i + 1, i_ptr)
      else
        Pointer.store(
          Pointer.load(Term.t(), Pointer.element_ptr(Term.t(), right_temp, j)),
          Pointer.element_ptr(Term.t(), arr, k)
        )

        Pointer.store(j + 1, j_ptr)
      end

      Pointer.store(k + 1, k_ptr)
    end

    while_loop(Pointer.load(i32(), i_ptr) < n1) do
      i = Pointer.load(i32(), i_ptr)
      k = Pointer.load(i32(), k_ptr)

      Pointer.store(
        Pointer.load(Term.t(), Pointer.element_ptr(Term.t(), left_temp, i)),
        Pointer.element_ptr(Term.t(), arr, k)
      )

      Pointer.store(i + 1, i_ptr)
      Pointer.store(k + 1, k_ptr)
    end

    while_loop(Pointer.load(i32(), j_ptr) < n2) do
      j = Pointer.load(i32(), j_ptr)
      k = Pointer.load(i32(), k_ptr)

      Pointer.store(
        Pointer.load(Term.t(), Pointer.element_ptr(Term.t(), right_temp, j)),
        Pointer.element_ptr(Term.t(), arr, k)
      )

      Pointer.store(j + 1, j_ptr)
      Pointer.store(k + 1, k_ptr)
    end

    op func.return() :: []
  end

  defm tim_sort(arr :: Pointer.t(), n :: i32()) do
    run_const = op arith.constant(value: Attribute.integer(i32(), 32)) :: i32()
    run = result_at(run_const, 0)
    i_ptr = Pointer.allocate(i32())
    zero_const = op arith.constant(value: Attribute.integer(i32(), 0)) :: i32()
    zero = result_at(zero_const, 0)
    Pointer.store(zero, i_ptr)

    while_loop(Pointer.load(i32(), i_ptr) < n) do
      i = Pointer.load(i32(), i_ptr)
      min = op arith.minsi(i + run - 1, n - 1) :: i32()
      min = result_at(min, 0)
      call insertion_sort(arr, i, min) :: []
      Pointer.store(i + run, i_ptr)
    end

    size_ptr = Pointer.allocate(i32())
    Pointer.store(run, size_ptr)

    while_loop(Pointer.load(i32(), size_ptr) < n) do
      size = Pointer.load(i32(), size_ptr)

      left_ptr = Pointer.allocate(i32())
      Pointer.store(zero, left_ptr)

      while_loop(Pointer.load(i32(), left_ptr) < n) do
        left = Pointer.load(i32(), left_ptr)
        mid = left + size - 1
        right = op arith.minsi(left + 2 * size - 1, n - 1) :: i32()
        right = result_at(right, 0)

        struct_if(mid < right) do
          call merge(arr, left, mid, right) :: []
        end

        Pointer.store(left + 2 * size, left_ptr)
      end

      Pointer.store(size * 2, size_ptr)
    end

    op func.return() :: []
  end

  defm copy_terms(env :: Env.t(), movable_list_ptr :: Pointer.t(), arr :: Pointer.t()) do
    head = Pointer.allocate(Term.t())
    zero_const = op arith.constant(value: Attribute.integer(i32(), 0)) :: i32()
    zero = result_at(zero_const, 0)
    i_ptr = Pointer.allocate(i32())
    Pointer.store(zero, i_ptr)

    while_loop(
      enif_get_list_cell(
        env,
        Pointer.load(Term.t(), movable_list_ptr),
        head,
        movable_list_ptr
      ) > 0
    ) do
      head_val = Pointer.load(Term.t(), head)
      i = Pointer.load(i32(), i_ptr)
      ith_term_ptr = Pointer.element_ptr(Term.t(), arr, i)
      Pointer.store(head_val, ith_term_ptr)
      Pointer.store(i + 1, i_ptr)
    end

    op func.return() :: []
  end

  defm sort(env, list, err) :: Term.t() do
    len_ptr = Pointer.allocate(i32())

    cond_br(enif_get_list_length(env, list, len_ptr) != 0) do
      movable_list_ptr = Pointer.allocate(Term.t())
      Pointer.store(list, movable_list_ptr)
      len = Pointer.load(i32(), len_ptr)
      arr = Pointer.allocate(Term.t(), len)
      call copy_terms(env, movable_list_ptr, arr) :: []
      call tim_sort(arr, len) :: []
      ret = enif_make_list_from_array(env, arr, len)
      op func.return(ret) :: []
    else
      op func.return(err) :: []
    end
  end
end
