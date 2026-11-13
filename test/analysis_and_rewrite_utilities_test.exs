defmodule AnalysisAndRewriteUtilitiesTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR.{Attribute, Block, DominanceInfo, IRMapping, IRRewriter, Location, OpOperand, Operation}
  alias Beaver.MLIR.{PostDominanceInfo, RewriterBase, Type, Value}

  defp control_flow_module(ctx) do
    "module {\n  func.func @branch(%cond: i1, %arg: i32) -> i32 {\n    cf.cond_br %cond, ^left, ^right\n  ^left:\n    %left_value = arith.addi %arg, %arg : i32\n    cf.br ^merge(%left_value : i32)\n  ^right:\n    %right_value = arith.subi %arg, %arg : i32\n    cf.br ^merge(%right_value : i32)\n  ^merge(%result: i32):\n    return %result : i32\n  ^unreachable:\n    return %arg : i32\n  }\n}\n"
    |> MLIR.Module.create!(ctx: ctx)
  end

  defp func_and_blocks(module) do
    func = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)
    region = func |> Beaver.Walker.regions() |> Enum.fetch!(0)
    {func, Enum.to_list(Beaver.Walker.blocks(region))}
  end

  test "dominance supports block, operation, and value queries", %{ctx: ctx} do
    module = control_flow_module(ctx)
    {func, [entry, left, right, merge, unreachable]} = func_and_blocks(module)
    [entry_branch] = Enum.to_list(Beaver.Walker.operations(entry))
    [left_add, left_branch] = Enum.to_list(Beaver.Walker.operations(left))
    [right_sub, _right_branch] = Enum.to_list(Beaver.Walker.operations(right))
    [merge_return] = Enum.to_list(Beaver.Walker.operations(merge))
    left_value = Operation.result(left_add, 0)

    assert :checked ==
             DominanceInfo.with_info(func, fn info ->
               assert DominanceInfo.dominates?(info, entry, entry)
               assert DominanceInfo.properly_dominates?(info, entry, merge)
               refute DominanceInfo.dominates?(info, left, right)
               assert DominanceInfo.dominates?(info, entry_branch, merge_return)
               assert DominanceInfo.properly_dominates?(info, left_add, left_branch)
               refute DominanceInfo.dominates?(info, left_add, right_sub)
               assert DominanceInfo.value_dominates?(info, left_value, left_branch)
               assert DominanceInfo.value_properly_dominates?(info, left_value, left_branch)
               refute DominanceInfo.value_dominates?(info, left_value, merge_return)
               assert MLIR.equal?(entry, DominanceInfo.nearest_common_dominator(info, left, right))
               assert DominanceInfo.reachable_from_entry?(info, merge)
               refute DominanceInfo.reachable_from_entry?(info, unreachable)
               assert ^info = DominanceInfo.invalidate(info)
               assert DominanceInfo.dominates?(info, entry, merge)
               :checked
             end)

    MLIR.Module.destroy(module)
  end

  test "post-dominance supports block and operation queries", %{ctx: ctx} do
    module = control_flow_module(ctx)
    {func, [entry, left, right, merge, _unreachable]} = func_and_blocks(module)
    [entry_branch] = Enum.to_list(Beaver.Walker.operations(entry))
    [left_add, _left_branch] = Enum.to_list(Beaver.Walker.operations(left))
    [merge_return] = Enum.to_list(Beaver.Walker.operations(merge))

    PostDominanceInfo.with_info(func, fn info ->
      assert PostDominanceInfo.post_dominates?(info, merge, left)
      assert PostDominanceInfo.properly_post_dominates?(info, merge, right)
      refute PostDominanceInfo.post_dominates?(info, left, entry)
      assert PostDominanceInfo.post_dominates?(info, merge_return, left_add)
      assert PostDominanceInfo.properly_post_dominates?(info, merge_return, entry_branch)
      assert ^info = PostDominanceInfo.invalidate(info)
      assert PostDominanceInfo.post_dominates?(info, merge, entry)
    end)

    MLIR.Module.destroy(module)
  end

  test "IRMapping maps all entity kinds and records clone correspondence", %{ctx: ctx} do
    module = control_flow_module(ctx)
    {func, [_entry | _]} = func_and_blocks(module)
    source_region = func |> Beaver.Walker.regions() |> Enum.fetch!(0)
    source_block = source_region |> Beaver.Walker.blocks() |> Enum.fetch!(0)
    source_argument = Block.get_arg!(source_block, 0)

    IRMapping.with_mapping(fn mapping ->
      clone = IRMapping.clone(mapping, func)

      try do
        clone_region = clone |> Beaver.Walker.regions() |> Enum.fetch!(0)
        clone_block = clone_region |> Beaver.Walker.blocks() |> Enum.fetch!(0)
        clone_argument = Block.get_arg!(clone_block, 0)
        assert MLIR.equal?(clone_block, IRMapping.lookup(mapping, source_block))
        assert MLIR.equal?(clone_argument, IRMapping.lookup(mapping, source_argument))
        assert IRMapping.contains?(mapping, source_block)
        assert IRMapping.contains?(mapping, source_argument)
        assert ^mapping = IRMapping.map(mapping, func, clone)
        assert MLIR.equal?(clone, IRMapping.lookup(mapping, func))
        assert IRMapping.contains?(mapping, func)
        assert ^mapping = IRMapping.erase(mapping, func)
        refute IRMapping.contains?(mapping, func)
        assert is_nil(IRMapping.lookup(mapping, func))
        assert MLIR.equal?(func, IRMapping.lookup_or_default(mapping, func))
        assert ^mapping = IRMapping.map(mapping, func, clone)
        assert ^mapping = IRMapping.clear(mapping)
        refute IRMapping.contains?(mapping, func)
        refute IRMapping.contains?(mapping, source_block)
        refute IRMapping.contains?(mapping, source_argument)
      after
        Operation.destroy(clone)
      end
    end)

    MLIR.Module.destroy(module)
  end

  test "structural equivalence can ignore locations without a text round-trip", %{ctx: ctx} do
    module = control_flow_module(ctx)
    {func, _blocks} = func_and_blocks(module)
    clone = Operation.clone(func)

    try do
      new_location = Location.file(name: "clone.mlir", line: 7, column: 11, ctx: ctx)
      assert ^clone = Operation.set_location(clone, new_location)
      refute Operation.equivalent?(func, clone)
      assert Operation.equivalent?(func, clone, ignore_locations: true)

      assert Operation.structural_hash(func, ignore_locations: true) ==
               Operation.structural_hash(clone, ignore_locations: true)

      tag = Attribute.integer(Type.i32(ctx: ctx), 1)
      :ok = Operation.put_discardable_attribute(clone, "test.tag", tag)
      refute Operation.equivalent?(func, clone, ignore_locations: true)
      assert Operation.equivalent?(func, clone, ignore_locations: true, ignore_discardable_attributes: true)

      assert Operation.structural_hash(func, ignore_locations: true, ignore_discardable_attributes: true) ==
               Operation.structural_hash(clone, ignore_locations: true, ignore_discardable_attributes: true)

      assert_raise ArgumentError, ~r/unsupported operation equivalence option/, fn ->
        Operation.equivalent?(func, clone, unknown: true)
      end
    after
      Operation.destroy(clone)
      MLIR.Module.destroy(module)
    end
  end

  test "structural equivalence exposes properties and commutativity flags", %{ctx: ctx} do
    module =
      "module {\n  func.func @operations(%lhs: i32, %rhs: i32) {\n    %0 = arith.addi %lhs, %rhs : i32\n    %1 = arith.addi %rhs, %lhs : i32\n    %2 = arith.constant 1 : i32\n    %3 = arith.constant 2 : i32\n    return\n  }\n}\n"
      |> MLIR.Module.create!(ctx: ctx)

    MLIR.Module.with_module(module, fn _ ->
      {func, _blocks} = func_and_blocks(module)

      [add, reversed_add, one, two, _return] =
        func
        |> Beaver.Walker.regions()
        |> Enum.fetch!(0)
        |> Beaver.Walker.blocks()
        |> Enum.fetch!(0)
        |> Beaver.Walker.operations()
        |> Enum.to_list()

      assert Operation.equivalent?(add, reversed_add, ignore_locations: true)
      refute Operation.equivalent?(add, reversed_add, ignore_locations: true, ignore_commutativity: true)
      refute Operation.equivalent?(one, two, ignore_locations: true)
      assert Operation.equivalent?(one, two, ignore_locations: true, ignore_properties: true)
    end)
  end

  test "conditional use replacement dispatches a safe Elixir predicate", %{ctx: ctx} do
    module =
      "module {\n  func.func @replace_selected_use(%arg: i32) -> i32 {\n    %replacement = arith.constant 7 : i32\n    %selected = arith.addi %arg, %replacement : i32\n    %kept = arith.subi %arg, %selected : i32\n    return %kept : i32\n  }\n}\n"
      |> MLIR.Module.create!(ctx: ctx)

    MLIR.Module.with_module(module, fn _ ->
      {func, _blocks} = func_and_blocks(module)
      block = func |> Beaver.Walker.regions() |> Enum.fetch!(0) |> Beaver.Walker.blocks() |> Enum.fetch!(0)
      argument = Block.get_arg!(block, 0)
      [replacement, selected, kept, _return] = block |> Beaver.Walker.operations() |> Enum.to_list()
      replacement_value = Operation.result(replacement, 0)
      caller = self()

      assert :ok =
               Value.replace_uses_with_if(argument, replacement_value, fn op_operand ->
                 assert self() == caller
                 assert OpOperand.operand_number(op_operand) == 0
                 assert MLIR.equal?(OpOperand.value(op_operand), argument)
                 OpOperand.owner(op_operand) |> Operation.name() == "arith.addi"
               end)

      [selected_lhs, _selected_rhs] = selected |> Beaver.Walker.operands() |> Enum.to_list()
      [kept_lhs, _kept_rhs] = kept |> Beaver.Walker.operands() |> Enum.to_list()
      assert MLIR.equal?(selected_lhs, replacement_value)
      assert MLIR.equal?(kept_lhs, argument)
    end)
  end

  test "conditional use replacement propagates predicate failures", %{ctx: ctx} do
    module = control_flow_module(ctx)
    {_func, [entry | _]} = func_and_blocks(module)
    argument = Block.get_arg!(entry, 1)

    MLIR.Module.with_module(module, fn _ ->
      assert_raise RuntimeError, "predicate failed", fn ->
        Value.replace_uses_with_if(argument, argument, fn _op_operand -> raise "predicate failed" end)
      end

      assert_raise ArgumentError, ~r/replacement predicate must return a boolean/, fn ->
        Value.replace_uses_with_if(argument, argument, fn _op_operand -> :replace end)
      end
    end)
  end

  defp constants_module(ctx) do
    "module {\n  func.func @constants() -> i32 {\n    %0 = arith.constant 1 : i32\n    %1 = arith.constant 2 : i32\n    return %1 : i32\n  }\n}\n"
    |> MLIR.Module.create!(ctx: ctx)
  end

  test "scoped insertion points restore on return and exception", %{ctx: ctx} do
    module = constants_module(ctx)
    {func, _blocks} = func_and_blocks(module)
    block = func |> Beaver.Walker.regions() |> Enum.fetch!(0) |> Beaver.Walker.blocks() |> Enum.fetch!(0)
    [first, second, _return] = Enum.to_list(Beaver.Walker.operations(block))

    MLIR.Module.with_module(module, fn _ ->
      IRRewriter.with_rewriter(second, fn rewriter ->
        assert MLIR.equal?(second, RewriterBase.operation_after_insertion(rewriter))

        assert :inside ==
                 RewriterBase.with_insertion_point(rewriter, {:start, block}, fn ->
                   assert MLIR.equal?(first, RewriterBase.operation_after_insertion(rewriter))
                   :inside
                 end)

        assert MLIR.equal?(second, RewriterBase.operation_after_insertion(rewriter))

        assert_raise RuntimeError, "rewrite failed", fn ->
          RewriterBase.with_insertion_point(rewriter, {:after, second}, fn -> raise "rewrite failed" end)
        end

        assert MLIR.equal?(second, RewriterBase.operation_after_insertion(rewriter))
      end)
    end)
  end

  test "saved insertion point round-trips through block creation and insertion", %{ctx: ctx} do
    module = constants_module(ctx)
    {_func, [block | _]} = func_and_blocks(module)
    [first | _] = Enum.to_list(Beaver.Walker.operations(block))

    MLIR.Module.with_module(module, fn _ ->
      IRRewriter.with_rewriter(first, fn rewriter ->
        assert MLIR.equal?(first, RewriterBase.operation_after_insertion(rewriter))
        saved = RewriterBase.save_insertion_point(rewriter)
        # Creating a block through the rewriter moves the insertion point to
        # the end of the freshly created block, which must not invalidate the
        # saved position.
        new_block =
          MLIR.CAPI.mlirRewriterBaseCreateBlockBefore(
            rewriter,
            block,
            0,
            Beaver.Native.array([], MLIR.Type),
            Beaver.Native.array([], MLIR.Location)
          )

        assert MLIR.null?(RewriterBase.operation_after_insertion(rewriter))

        constant =
          %Beaver.Changeset{name: "arith.constant", context: ctx}
          |> Beaver.Changeset.add_argument(value: Attribute.integer(Type.i32(), 3))
          |> Beaver.Changeset.add_result(Type.i32())
          |> Operation.create()

        RewriterBase.insert(rewriter, constant)
        assert [inserted] = Enum.to_list(Beaver.Walker.operations(new_block))
        assert MLIR.equal?(constant, inserted)
        RewriterBase.restore_insertion_point(rewriter, saved)
        # The insertion point is back exactly where it was saved.
        assert MLIR.equal?(first, RewriterBase.operation_after_insertion(rewriter))
      end)
    end)
  end
end
