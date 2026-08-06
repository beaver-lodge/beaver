defmodule NativePatternTest do
  use Beaver.Case, async: true
  use Beaver

  import ExUnit.CaptureLog

  alias Beaver.MLIR.Dialect.{Arith, Func}
  alias Beaver.Pattern.Native
  require Native
  require Func

  defmodule TestNativePatterns do
    use Beaver
    use Beaver.Pattern.Native

    alias Beaver.MLIR.Dialect.Arith

    defrewrite fold_const_add(operation, rewriter, _state),
      root: Arith.addi(),
      operands: [lhs, rhs],
      results: [_result],
      attributes: [],
      benefit: 2 do
      base = MLIR.PatternRewriter.as_base(rewriter)
      ctx = MLIR.context(operation)

      with {:ok, lhs_operation} <- MLIR.Value.owner(lhs),
           {:ok, rhs_operation} <- MLIR.Value.owner(rhs),
           "arith.constant" <- MLIR.Operation.name(lhs_operation),
           "arith.constant" <- MLIR.Operation.name(rhs_operation),
           {:ok, lhs_value_attribute} <- MLIR.Operation.fetch(lhs_operation, "value"),
           {:ok, rhs_value_attribute} <- MLIR.Operation.fetch(rhs_operation, "value") do
        lhs_value = MLIR.Attribute.value(lhs_value_attribute)
        rhs_value = MLIR.Attribute.value(rhs_value_attribute)

        mlir ctx: ctx, ip: base do
          replacement =
            Arith.constant(value: Attribute.integer(Type.i64(), lhs_value + rhs_value)) >>>
              Type.i64()

          MLIR.RewriterBase.replace_op(base, operation, [replacement])
        end

        :ok
      else
        _ -> :no_match
      end
    end

    defrewrite match_with_attrs(operation, rewriter, state),
      root: Arith.constant(),
      operands: [],
      results: [_result],
      attributes: [value: value_attribute] do
      {test_pid, count} = state

      if MLIR.Attribute.value(value_attribute) == 300 do
        base = MLIR.PatternRewriter.as_base(rewriter)
        ctx = MLIR.context(operation)

        mlir ctx: ctx, ip: base do
          replacement =
            Arith.constant(value: Attribute.integer(Type.i64(), 999)) >>> Type.i64()

          MLIR.RewriterBase.replace_op(base, operation, [replacement])
        end

        {:ok, {test_pid, count + 1}}
      else
        {:error, {test_pid, count}}
      end
    end

    defrewrite invalid_outcome(_operation, _rewriter, _state),
      root: Arith.constant() do
      :invalid_outcome_value
    end

    defrewrite raise_in_callback(_operation, _rewriter, state),
      root: Arith.addi() do
      raise "native rewrite boom: #{inspect(state)}"
    end

    defrewrite probe_borrowed_rewriter(_operation, rewriter, state),
      root: Arith.addi() do
      {test_pid, count} = state
      %MLIR.PatternRewriter{} = rewriter
      %MLIR.RewriterBase{} = MLIR.PatternRewriter.as_base(rewriter)
      send(test_pid, {:rewriter_usable_in_callback, self()})
      {:error, {test_pid, count + 1}}
    end
  end

  defmodule MismatchPatterns do
    use Beaver.Pattern.Native

    defrewrite expect_three_operands(_operation, _rewriter, test_pid),
      root: "arith.addi",
      operands: [_a, _b, _c] do
      send(test_pid, :unexpected_operand_match)
      :no_match
    end

    defrewrite expect_missing_attribute(_operation, _rewriter, test_pid),
      root: "arith.addi",
      attributes: [non_existent_attribute_xyz: _value] do
      send(test_pid, :unexpected_attribute_match)
      :no_match
    end
  end

  defmodule StateTracker do
    def construct(state), do: state

    def destruct({test_pid, final_state}) do
      send(test_pid, {:destruct_state, final_state})
    end
  end

  defp create_add_ir(ctx, lhs, rhs) do
    mlir ctx: ctx do
      module do
        Func.func test_func(function_type: Type.function([], [Type.i64()])) do
          region do
            block do
              lhs = Arith.constant(value: Attribute.integer(Type.i64(), lhs)) >>> Type.i64()
              rhs = Arith.constant(value: Attribute.integer(Type.i64(), rhs)) >>> Type.i64()
              result = Arith.addi(lhs, rhs) >>> Type.i64()
              Func.return(result) >>> []
            end
          end
        end
      end
    end
    |> MLIR.verify!()
  end

  test "descriptor defaults, validation, and builder options" do
    descriptor = TestNativePatterns.fold_const_add()

    assert %Beaver.Pattern.Native.Descriptor{} = descriptor
    assert descriptor.name == :fold_const_add
    assert descriptor.root == "arith.addi"
    assert descriptor.benefit == 2
    assert descriptor.init_state == nil

    overridden = TestNativePatterns.fold_const_add(benefit: 10, init_state: :custom)
    assert overridden.benefit == 10
    assert overridden.init_state == :custom

    assert_raise ArgumentError, ~r/unsupported native rewrite builder options/, fn ->
      TestNativePatterns.fold_const_add(unknown: true)
    end

    assert_raise ArgumentError, ~r/non-negative integer/, fn ->
      TestNativePatterns.fold_const_add(benefit: -1)
    end
  end

  test "invalid declarations fail while the macro is expanded" do
    missing_root =
      quote do
        Native.defrewrite broken(operation, rewriter, state) do
          {operation, rewriter, state}
        end
      end

    assert_raise ArgumentError, ~r/requires the :root option/, fn ->
      Macro.expand_once(missing_root, __ENV__)
    end

    invalid_operands =
      quote do
        Native.defrewrite broken(operation, rewriter, state),
          root: "arith.addi",
          operands: :not_a_list do
          {operation, rewriter, state}
        end
      end

    assert_raise ArgumentError, ~r/:operands must be a literal list pattern/, fn ->
      Macro.expand_once(invalid_operands, __ENV__)
    end

    destructured_argument =
      quote do
        Native.defrewrite broken(operation, rewriter, {_pid, _count}),
          root: "arith.addi" do
          :no_match
        end
      end

    assert_raise ArgumentError, ~r/callback arguments must be variables/, fn ->
      Macro.expand_once(destructured_argument, __ENV__)
    end
  end

  test "binds operands and results and rewrites through the native callback", %{ctx: ctx} do
    ir = create_add_ir(ctx, 10, 20)

    MLIR.Rewrite.apply_patterns!(ir, [TestNativePatterns.fold_const_add()],
      enable_folding: false,
      enable_constant_cse: false
    )

    assert Enum.any?(function_operations(ir), fn operation ->
             if MLIR.Operation.name(operation) == Arith.constant() do
               {:ok, attribute} = MLIR.Operation.fetch(operation, "value")
               MLIR.Attribute.value(attribute) == 30
             else
               false
             end
           end)

    MLIR.verify!(ir)
  end

  test "operand and missing-attribute mismatches are no-match", %{ctx: ctx} do
    ir = create_add_ir(ctx, 5, 15)

    MLIR.Rewrite.apply_patterns!(
      ir,
      [
        MismatchPatterns.expect_three_operands(init_state: self()),
        MismatchPatterns.expect_missing_attribute(init_state: self())
      ],
      enable_folding: false,
      enable_constant_cse: false
    )

    assert Enum.any?(function_operations(ir), &(MLIR.Operation.name(&1) == Arith.addi()))
    refute_received :unexpected_operand_match
    refute_received :unexpected_attribute_match
    MLIR.verify!(ir)
  end

  test "construct, state transitions, and destruct use the existing lifecycle", %{ctx: ctx} do
    ir = create_add_ir(ctx, 100, 200)
    set = MLIR.RewritePatternSet.create(ctx)

    descriptor =
      TestNativePatterns.match_with_attrs(
        init_state: {self(), 0},
        construct: &StateTracker.construct/1,
        destruct: &StateTracker.destruct/1
      )

    MLIR.RewritePatternSet.add(set, descriptor, ctx: ctx)

    frozen_set = MLIR.RewritePatternSet.freeze(set)
    MLIR.Rewrite.apply_patterns!(ir, frozen_set)
    MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen_set)

    assert_received {:destruct_state, 1}
    MLIR.verify!(ir)
  end

  test "callback exceptions become native no-match failures", %{ctx: ctx} do
    ir = create_add_ir(ctx, 1, 2)

    log =
      capture_log(fn ->
        MLIR.Rewrite.apply_patterns!(
          ir,
          [TestNativePatterns.raise_in_callback(init_state: :test_state)],
          enable_folding: false,
          enable_constant_cse: false
        )
      end)

    assert log =~ "native rewrite boom: :test_state"
    assert Enum.any?(function_operations(ir), &(MLIR.Operation.name(&1) == Arith.addi()))
    MLIR.verify!(ir)
  end

  test "borrowed rewriter is used during callback work without escaping", %{ctx: ctx} do
    ir = create_add_ir(ctx, 3, 4)

    descriptor =
      TestNativePatterns.probe_borrowed_rewriter(
        init_state: {self(), 0},
        destruct: fn {test_pid, count} -> send(test_pid, {:probe_destruct, count}) end
      )

    MLIR.Rewrite.apply_patterns!(ir, [descriptor],
      enable_folding: false,
      enable_constant_cse: false
    )

    assert_received {:rewriter_usable_in_callback, callback_pid}
    refute callback_pid == self()
    assert_received {:probe_destruct, 1}
  end

  test "unsupported outcomes raise before crossing the native boundary", %{ctx: ctx} do
    descriptor = TestNativePatterns.invalid_outcome()
    ir = create_add_ir(ctx, 1, 2)
    constant = Enum.find(function_operations(ir), &(MLIR.Operation.name(&1) == Arith.constant()))

    assert_raise RuntimeError, ~r/unsupported outcome/, fn ->
      descriptor.match_and_rewrite.(nil, constant, nil, :initial_state)
    end
  end

  test "a descriptor can be added directly to a rewrite pattern set", %{ctx: ctx} do
    ir = create_add_ir(ctx, 7, 8)
    set = MLIR.RewritePatternSet.create(ctx)

    MLIR.RewritePatternSet.add(set, TestNativePatterns.fold_const_add(), ctx: ctx)

    MLIR.Rewrite.apply_patterns!(ir, set,
      enable_folding: false,
      enable_constant_cse: false
    )

    assert Enum.any?(function_operations(ir), fn operation ->
             if MLIR.Operation.name(operation) == Arith.constant() do
               {:ok, attribute} = MLIR.Operation.fetch(operation, "value")
               MLIR.Attribute.value(attribute) == 15
             else
               false
             end
           end)

    MLIR.verify!(ir)
  end

  defp function_operations(ir) do
    ir
    |> MLIR.Module.body()
    |> Beaver.Walker.operations()
    |> Enum.at(0)
    |> MLIR.Dialect.Func.entry_block()
    |> Beaver.Walker.operations()
    |> Enum.to_list()
  end
end
