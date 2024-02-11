op ex.add()["other attribute": 42 :: i64] ::
     {i64, i32} do
  _ ->
    op "erl.what", [], [attr1: 41 :: i64], i1
end

var = op dialect.op1()[attribute_name: 42 :: i32] :: {i1, i16, i32, i64}

op dialect.op2()["other attribute": 42 :: i64] ::
     {i64, i1} do
  _ ->
    op dialect.inner_op2() :: {}
    op dialect.inner_op3(var._0, var._2, var._3) :: {}
    br bb2(var._0)

  bb1(arg :: i32) ->
    op dialect.inner_op4(arg)[attribute_name2: 42 :: i32] :: {}
    op "dialect.inner_op4-1", [], [], {}

  bb2(arg2 :: i1) ->
    op dialect.inner_op5(arg2) :: {}
    op "dialect.inner_op5-1", [], [], {}
    br bb1(var._2)
end
