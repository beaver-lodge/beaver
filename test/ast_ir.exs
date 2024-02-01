op ex.add()["other attribute": 42 :: i64] ::
     {i64, i32} do
  _ ->
    op "erl.what", [], [attr1: 41 :: i64], i1
end

var = op dialect.op1()[attribute_name: 42 :: i32] :: {i1, i16, i32, i64}

op dialect.op2()["other attribute": 42 :: i64] ::
     {i64, i1} do
  _ ->
    op dialect.innerop2() :: {}
    op dialect.innerop3(var._0, var._2, var._3) :: {}
    jump bb1(var._0)

  bb1(arg :: i32) ->
    op dialect.innerop4(arg)[attribute_name2: 42 :: i32] :: {}
    op "dialect.innerop4-1", [], [], {}

  bb2(arg2 :: i64) ->
    op dialect.innerop5(arg2) :: {}
    op "dialect.innerop5-1", [], [], {}
end
