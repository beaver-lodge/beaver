defm some_func2() do
  v0 = 0
  cond0 = true

  if cond0 do
    v1 = 0
    v1 + v1
  else
    v2 = 0
    v0 + v2
  end
end

Func.func some_func2(function_type: Type.function([], [Type.i(32)])) do
  region do
    block bb_entry() do
      v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) when is(v0, Type.i(32))
      cond0 = Arith.constant(true) when is(cond0, Type.i(32))

      cond_br(cond0) do
        bb1()
      else
        bb2(v0)
      end
    end

    block bb1() do
      v1 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) when is(v1, Type.i(32))
      add = Arith.addi(v0, v0) when is(add, Type.i(1))
      bb2(v1)
    end

    block bb2(arg) when is(arg, Type.i(32)) do
      v2 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) when is(v2, Type.i(32))
      add = Arith.addi(arg, v2) when is(add, Type.i(32))
      Func.return(add)
    end
  end
end
