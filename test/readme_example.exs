Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
  region do
    block _bb_entry() do
      v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
      cond0 = Arith.constant(true) >>> Type.i(1)
      CF.cond_br(cond0, Beaver.Env.block(bb1), {Beaver.Env.block(bb2), [v0]}) >>> []
    end

    block bb1() do
      v1 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
      _add = Arith.addi(v0, v0) >>> Type.i(32)
      CF.br({Beaver.Env.block(bb2), [v1]}) >>> []
    end

    block bb2(arg >>> Type.i(32)) do
      v2 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
      add = Arith.addi(arg, v2) >>> Type.i(32)
      Func.return(add) >>> []
    end
  end
end
|> MLIR.Operation.verify!(debug: true)
