defmodule Beaver.ConversionHelpers do
  @moduledoc false
  alias Beaver.MLIR

  def materialization_module(ctx, kind) do
    source =
      case kind do
        :source ->
          ~S[module { %a = arith.constant 1 : i64 %s = "foo.source"() : () -> i32 "foo.keep"(%s) : (i32) -> () }]

        :target ->
          ~S[module { %a = "foo.producer"() : () -> i32 "foo.sink"(%a) : (i32) -> () }]

        :one_to_n ->
          ~S[module { %a = arith.constant 1 : i32 "foo.sink"(%a) : (i32) -> () }]
      end

    MLIR.Module.create!(source, ctx: ctx)
  end

  def insert(rewriter, name, location, arguments, results \\ []) do
    operation =
      %Beaver.Changeset{name: name, context: MLIR.context(location), location: location}
      |> Beaver.Changeset.add_argument(arguments)
      |> Beaver.Changeset.add_result(results)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(rewriter, operation)
    operation
  end

  def build_unrealized_cast(rewriter, output_types, inputs, location) do
    insert(rewriter, "builtin.unrealized_conversion_cast", location, inputs, output_types)
    |> MLIR.Operation.results()
    |> Enum.to_list()
  end
end
