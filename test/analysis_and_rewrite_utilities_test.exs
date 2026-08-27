defmodule AnalysisAndRewriteUtilitiesTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR

  defp control_flow_module(ctx) do
    MLIR.Module.create!(
      """
      module {
        func.func @branch(%cond: i1, %arg: i32) -> i32 {
          cf.cond_br %cond, ^left, ^right
        ^left:
          %left_value = arith.addi %arg, %arg : i32
          cf.br ^merge(%left_value : i32)
        ^right:
          %right_value = arith.subi %arg, %arg : i32
          cf.br ^merge(%right_value : i32)
        ^merge(%result: i32):
          return %result : i32
        ^unreachable:
          return %arg : i32
        }
      }
      """,
      ctx: ctx
    )
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
    left_value = MLIR.Operation.result(left_add, 0)

    assert :checked ==
             MLIR.DominanceInfo.with_info(func, fn info ->
               assert MLIR.DominanceInfo.dominates?(info, entry, entry)
               assert MLIR.DominanceInfo.properly_dominates?(info, entry, merge)
               refute MLIR.DominanceInfo.dominates?(info, left, right)

               assert MLIR.DominanceInfo.dominates?(info, entry_branch, merge_return)
               assert MLIR.DominanceInfo.properly_dominates?(info, left_add, left_branch)
               refute MLIR.DominanceInfo.dominates?(info, left_add, right_sub)

               assert MLIR.DominanceInfo.value_dominates?(info, left_value, left_branch)

               assert MLIR.DominanceInfo.value_properly_dominates?(
                        info,
                        left_value,
                        left_branch
                      )

               refute MLIR.DominanceInfo.value_dominates?(info, left_value, merge_return)

               assert MLIR.equal?(
                        entry,
                        MLIR.DominanceInfo.nearest_common_dominator(info, left, right)
                      )

               assert MLIR.DominanceInfo.reachable_from_entry?(info, merge)
               refute MLIR.DominanceInfo.reachable_from_entry?(info, unreachable)

               assert ^info = MLIR.DominanceInfo.invalidate(info)
               assert MLIR.DominanceInfo.dominates?(info, entry, merge)
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

    MLIR.PostDominanceInfo.with_info(func, fn info ->
      assert MLIR.PostDominanceInfo.post_dominates?(info, merge, left)
      assert MLIR.PostDominanceInfo.properly_post_dominates?(info, merge, right)
      refute MLIR.PostDominanceInfo.post_dominates?(info, left, entry)

      assert MLIR.PostDominanceInfo.post_dominates?(info, merge_return, left_add)

      assert MLIR.PostDominanceInfo.properly_post_dominates?(
               info,
               merge_return,
               entry_branch
             )

      assert ^info = MLIR.PostDominanceInfo.invalidate(info)
      assert MLIR.PostDominanceInfo.post_dominates?(info, merge, entry)
    end)

    MLIR.Module.destroy(module)
  end

  test "IRMapping maps all entity kinds and records clone correspondence", %{ctx: ctx} do
    module = control_flow_module(ctx)
    {func, [_entry | _]} = func_and_blocks(module)
    source_region = func |> Beaver.Walker.regions() |> Enum.fetch!(0)
    source_block = source_region |> Beaver.Walker.blocks() |> Enum.fetch!(0)
    source_argument = MLIR.Block.get_arg!(source_block, 0)

    MLIR.IRMapping.with_mapping(fn mapping ->
      clone = MLIR.IRMapping.clone(mapping, func)

      try do
        clone_region = clone |> Beaver.Walker.regions() |> Enum.fetch!(0)
        clone_block = clone_region |> Beaver.Walker.blocks() |> Enum.fetch!(0)
        clone_argument = MLIR.Block.get_arg!(clone_block, 0)

        assert MLIR.equal?(clone_block, MLIR.IRMapping.lookup(mapping, source_block))
        assert MLIR.equal?(clone_argument, MLIR.IRMapping.lookup(mapping, source_argument))
        assert MLIR.IRMapping.contains?(mapping, source_block)
        assert MLIR.IRMapping.contains?(mapping, source_argument)

        assert ^mapping = MLIR.IRMapping.map(mapping, func, clone)
        assert MLIR.equal?(clone, MLIR.IRMapping.lookup(mapping, func))
        assert MLIR.IRMapping.contains?(mapping, func)

        assert ^mapping = MLIR.IRMapping.erase(mapping, func)
        refute MLIR.IRMapping.contains?(mapping, func)
        assert is_nil(MLIR.IRMapping.lookup(mapping, func))
        assert MLIR.equal?(func, MLIR.IRMapping.lookup_or_default(mapping, func))

        assert ^mapping = MLIR.IRMapping.clear(mapping)
        refute MLIR.IRMapping.contains?(mapping, source_block)
        refute MLIR.IRMapping.contains?(mapping, source_argument)
      after
        MLIR.Operation.destroy(clone)
      end
    end)

    MLIR.Module.destroy(module)
  end

  test "structural equivalence can ignore locations without a text round-trip", %{ctx: ctx} do
    module = control_flow_module(ctx)
    {func, _blocks} = func_and_blocks(module)
    clone = MLIR.Operation.clone(func)

    try do
      new_location = MLIR.Location.file(name: "clone.mlir", line: 7, column: 11, ctx: ctx)
      assert ^clone = MLIR.Operation.set_location(clone, new_location)

      refute MLIR.Operation.equivalent?(func, clone)
      assert MLIR.Operation.equivalent?(func, clone, ignore_locations: true)

      assert MLIR.Operation.structural_hash(func, ignore_locations: true) ==
               MLIR.Operation.structural_hash(clone, ignore_locations: true)

      tag = MLIR.Attribute.integer(MLIR.Type.i32(ctx: ctx), 1)

      :ok = MLIR.Operation.put_discardable_attribute(clone, "test.tag", tag)

      refute MLIR.Operation.equivalent?(func, clone, ignore_locations: true)

      assert MLIR.Operation.equivalent?(func, clone,
               ignore_locations: true,
               ignore_discardable_attributes: true
             )

      assert MLIR.Operation.structural_hash(
               func,
               ignore_locations: true,
               ignore_discardable_attributes: true
             ) ==
               MLIR.Operation.structural_hash(
                 clone,
                 ignore_locations: true,
                 ignore_discardable_attributes: true
               )

      assert_raise ArgumentError, ~r/unsupported operation equivalence option/, fn ->
        MLIR.Operation.equivalent?(func, clone, unknown: true)
      end
    after
      MLIR.Operation.destroy(clone)
      MLIR.Module.destroy(module)
    end
  end

  test "structural equivalence exposes properties and commutativity flags", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        module {
          func.func @operations(%lhs: i32, %rhs: i32) {
            %0 = arith.addi %lhs, %rhs : i32
            %1 = arith.addi %rhs, %lhs : i32
            %2 = arith.constant 1 : i32
            %3 = arith.constant 2 : i32
            return
          }
        }
        """,
        ctx: ctx
      )

    try do
      {func, _blocks} = func_and_blocks(module)

      [add, reversed_add, one, two, _return] =
        func
        |> Beaver.Walker.regions()
        |> Enum.fetch!(0)
        |> Beaver.Walker.blocks()
        |> Enum.fetch!(0)
        |> Beaver.Walker.operations()
        |> Enum.to_list()

      assert MLIR.Operation.equivalent?(add, reversed_add, ignore_locations: true)

      refute MLIR.Operation.equivalent?(add, reversed_add,
               ignore_locations: true,
               ignore_commutativity: true
             )

      refute MLIR.Operation.equivalent?(one, two, ignore_locations: true)

      assert MLIR.Operation.equivalent?(one, two,
               ignore_locations: true,
               ignore_properties: true
             )
    after
      MLIR.Module.destroy(module)
    end
  end

  test "conditional use replacement dispatches a safe Elixir predicate", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        module {
          func.func @replace_selected_use(%arg: i32) -> i32 {
            %replacement = arith.constant 7 : i32
            %selected = arith.addi %arg, %replacement : i32
            %kept = arith.subi %arg, %selected : i32
            return %kept : i32
          }
        }
        """,
        ctx: ctx
      )

    try do
      {func, _blocks} = func_and_blocks(module)

      block =
        func
        |> Beaver.Walker.regions()
        |> Enum.fetch!(0)
        |> Beaver.Walker.blocks()
        |> Enum.fetch!(0)

      argument = MLIR.Block.get_arg!(block, 0)

      [replacement, selected, kept, _return] =
        block |> Beaver.Walker.operations() |> Enum.to_list()

      replacement_value = MLIR.Operation.result(replacement, 0)
      caller = self()

      assert :ok =
               MLIR.Value.replace_uses_with_if(argument, replacement_value, fn op_operand ->
                 assert self() == caller
                 assert MLIR.OpOperand.operand_number(op_operand) == 0
                 assert MLIR.equal?(MLIR.OpOperand.value(op_operand), argument)
                 MLIR.OpOperand.owner(op_operand) |> MLIR.Operation.name() == "arith.addi"
               end)

      [selected_lhs, _selected_rhs] = selected |> Beaver.Walker.operands() |> Enum.to_list()
      [kept_lhs, _kept_rhs] = kept |> Beaver.Walker.operands() |> Enum.to_list()

      assert MLIR.equal?(selected_lhs, replacement_value)
      assert MLIR.equal?(kept_lhs, argument)
    after
      MLIR.Module.destroy(module)
    end
  end

  test "conditional use replacement propagates predicate failures", %{ctx: ctx} do
    module = control_flow_module(ctx)
    {_func, [entry | _]} = func_and_blocks(module)
    argument = MLIR.Block.get_arg!(entry, 1)

    try do
      assert_raise RuntimeError, "predicate failed", fn ->
        MLIR.Value.replace_uses_with_if(argument, argument, fn _op_operand ->
          raise "predicate failed"
        end)
      end

      assert_raise ArgumentError, ~r/replacement predicate must return a boolean/, fn ->
        MLIR.Value.replace_uses_with_if(argument, argument, fn _op_operand -> :replace end)
      end
    after
      MLIR.Module.destroy(module)
    end
  end

  test "scoped insertion points restore on return and exception", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        module {
          func.func @constants() -> i32 {
            %0 = arith.constant 1 : i32
            %1 = arith.constant 2 : i32
            return %1 : i32
          }
        }
        """,
        ctx: ctx
      )

    {func, _blocks} = func_and_blocks(module)

    block =
      func
      |> Beaver.Walker.regions()
      |> Enum.fetch!(0)
      |> Beaver.Walker.blocks()
      |> Enum.fetch!(0)

    [first, second, _return] = Enum.to_list(Beaver.Walker.operations(block))

    try do
      MLIR.IRRewriter.with_rewriter(second, fn rewriter ->
        assert MLIR.equal?(second, MLIR.RewriterBase.operation_after_insertion(rewriter))

        assert :inside ==
                 MLIR.RewriterBase.with_insertion_point(rewriter, {:start, block}, fn ->
                   assert MLIR.equal?(
                            first,
                            MLIR.RewriterBase.operation_after_insertion(rewriter)
                          )

                   :inside
                 end)

        assert MLIR.equal?(second, MLIR.RewriterBase.operation_after_insertion(rewriter))

        assert_raise RuntimeError, "rewrite failed", fn ->
          MLIR.RewriterBase.with_insertion_point(rewriter, {:after, second}, fn ->
            raise "rewrite failed"
          end)
        end

        assert MLIR.equal?(second, MLIR.RewriterBase.operation_after_insertion(rewriter))
      end)
    after
      MLIR.Module.destroy(module)
    end
  end

  test "saved insertion point round-trips through block creation and insertion", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        module {
          func.func @constants() -> i32 {
            %0 = arith.constant 1 : i32
            %1 = arith.constant 2 : i32
            return %1 : i32
          }
        }
        """,
        ctx: ctx
      )

    {_func, [block | _]} = func_and_blocks(module)
    [first | _] = Enum.to_list(Beaver.Walker.operations(block))

    try do
      MLIR.IRRewriter.with_rewriter(first, fn rewriter ->
        assert MLIR.equal?(first, MLIR.RewriterBase.operation_after_insertion(rewriter))

        saved = MLIR.RewriterBase.save_insertion_point(rewriter)

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

        assert MLIR.null?(MLIR.RewriterBase.operation_after_insertion(rewriter))

        constant =
          %Beaver.Changeset{name: "arith.constant", context: ctx}
          |> Beaver.Changeset.add_argument(value: MLIR.Attribute.integer(MLIR.Type.i32(), 3))
          |> Beaver.Changeset.add_result(MLIR.Type.i32())
          |> MLIR.Operation.create()

        MLIR.RewriterBase.insert(rewriter, constant)

        assert [inserted] = Enum.to_list(Beaver.Walker.operations(new_block))
        assert MLIR.equal?(constant, inserted)

        MLIR.RewriterBase.restore_insertion_point(rewriter, saved)

        # The insertion point is back exactly where it was saved.
        assert MLIR.equal?(first, MLIR.RewriterBase.operation_after_insertion(rewriter))
      end)
    after
      MLIR.Module.destroy(module)
    end
  end
end
