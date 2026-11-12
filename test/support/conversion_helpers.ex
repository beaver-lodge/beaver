defmodule Beaver.ConversionHelpers do
  @moduledoc false
  alias Beaver.MLIR
  alias MLIR.{ConversionPatternRewriter, Operation, Type, Value}

  @sources %{
    source: ~S[module { %a = arith.constant 1 : i64 %s = "foo.source"() : () -> i32 "foo.keep"(%s) : (i32) -> () }],
    target: ~S[module { %a = "foo.producer"() : () -> i32 "foo.sink"(%a) : (i32) -> () }],
    one_to_n: ~S[module { %a = arith.constant 1 : i32 "foo.sink"(%a) : (i32) -> () }]
  }
  def materialization_module(ctx, kind), do: MLIR.Module.create!(Map.fetch!(@sources, kind), ctx: ctx)

  # The raw conversion API and Plan must exercise the same callbacks and IR.
  def materialization_fixture(ctx, kind) do
    owner = self()
    module = materialization_module(ctx, kind)
    i32 = Type.i32(ctx: ctx)
    i64 = Type.i64(ctx: ctx)

    convert = conversion_callback(kind, owner, i32, i64)
    materialize = materialization_callback(kind, owner)
    rewrite = rewrite_callback(kind, owner, module, i64)

    {conversion, materialization, callback} =
      case kind do
        :source -> {:conversion, :source_materialization, &materialize.(&1, &2, &3, &4, nil)}
        :target -> {:conversion, :target_materialization, materialize}
        :one_to_n -> {:one_to_n, :one_to_n_target_materialization, materialize}
      end

    %{
      module: module,
      callbacks: [{conversion, convert}, {materialization, callback}],
      rewrite: rewrite,
      root: if(kind == :source, do: "foo.source", else: "foo.sink"),
      legal_ops: %{source: ["foo.keep"], target: ~w(foo.producer foo.sink_legal), one_to_n: []}[kind],
      dialects: if(kind == :target, do: ["builtin"], else: ~w(builtin arith)),
      pattern_opts: [one_to_n: kind == :one_to_n]
    }
  end

  defp conversion_callback(kind, owner, i32, i64) do
    converted = if kind == :one_to_n, do: [i64, i64], else: i64

    fn type ->
      if kind == :target, do: send(owner, {:target_conversion, MLIR.to_string(type)})

      cond do
        MLIR.equal?(type, i32) -> converted
        kind == :target -> type
        true -> :declined
      end
    end
  end

  defp materialization_callback(kind, owner) do
    fn rw, types, inputs, loc, original ->
      send(owner, {kind, Enum.map(List.wrap(types), &MLIR.to_string/1), length(inputs)})
      if kind == :target, do: send(owner, {:original_type, MLIR.to_string(original)})
      values = build_unrealized_cast(rw, List.wrap(types), inputs, loc)
      if kind == :one_to_n, do: values, else: hd(values)
    end
  end

  defp rewrite_callback(kind, owner, module, i64) do
    fn operation, operands, rw ->
      base = ConversionPatternRewriter.as_base(rw)

      case {kind, operands} do
        {:source, []} ->
          constant = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)
          ConversionPatternRewriter.replace_op(rw, operation, Operation.result(constant, 0))

        {:target, [converted]} ->
          send(owner, {:target_operand, MLIR.to_string(Value.type(converted))})
          insert(base, "foo.sink_legal", Operation.location(operation), converted)
          ConversionPatternRewriter.erase_op(rw, operation)

        {:one_to_n, [[lhs, rhs]]} ->
          insert(base, "arith.addi", Operation.location(operation), [lhs, rhs], i64)
          ConversionPatternRewriter.erase_op(rw, operation)
      end

      :ok
    end
  end

  def assert_materialized(module, kind) do
    import ExUnit.Assertions
    expected = {kind, %{source: ["i32"], target: ["i64"], one_to_n: ["i64", "i64"]}[kind], 1}
    assert_receive ^expected
    assert MLIR.to_string(module) =~ "unrealized_conversion_cast"

    if kind == :one_to_n do
      assert MLIR.to_string(module) =~ "arith.addi"
      refute MLIR.to_string(module) =~ "foo.sink"
    end

    if kind == :target do
      assert_receive {:target_conversion, "i32"}
      assert_receive {:target_operand, "i64"}
      assert_receive {:original_type, "i32"}
    end
  end

  def insert(rewriter, name, location, arguments, results \\ []),
    do: MLIR.RewriterBase.emit_op(rewriter, location, name, arguments, results)

  def build_unrealized_cast(rewriter, output_types, inputs, location) do
    insert(rewriter, "builtin.unrealized_conversion_cast", location, inputs, output_types)
    |> MLIR.Operation.results()
    |> Enum.to_list()
  end
end
