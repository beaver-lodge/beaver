defmodule Beaver.MLIR.CounterPlanTest do
  use Beaver.Slang, name: "counter_plan_test"

  alias Beaver.MLIR.Type

  defop inc(value = Type.i32()), do: [Type.i32()]
end

defmodule Beaver.MLIR.ConversionPlanNativePatterns do
  use Beaver.Pattern.Native

  alias Beaver.MLIR

  defrewrite erase_drop(operation, rewriter, owner), root: "plan_native.drop" do
    send(owner, :native_plan_rewrite)
    rewriter |> MLIR.PatternRewriter.as_base() |> MLIR.RewriterBase.erase_op(operation)
    :ok
  end
end

defmodule Beaver.MLIR.ConversionPlanTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.ConversionPlanNativePatterns

  test "lowers a Slang dialect using Plan and reuses the Plan across fresh MLIR contexts" do
    plan =
      Plan.new(
        mode: :full,
        folding_mode: :after_patterns,
        build_materializations: true,
        timeout: 5_000
      )
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_legal_dialect("func")
      |> Plan.add_legal_dialect("arith")
      |> Plan.add_illegal_dialect("counter_plan_test")
      |> Plan.add_conversion(fn type -> type end, version: "1.0")
      |> Plan.add_conversion_pattern(
        "counter_plan_test.inc",
        fn operation, [input], rewriter ->
          context = MLIR.context(operation)
          base = MLIR.ConversionPatternRewriter.as_base(rewriter)
          type = MLIR.Value.type(input)
          location = MLIR.Operation.location(operation)

          one =
            %Beaver.Changeset{name: "arith.constant", context: context, location: location}
            |> Beaver.Changeset.add_argument(value: MLIR.Attribute.integer(type, 1))
            |> Beaver.Changeset.add_result(type)
            |> MLIR.Operation.create()

          add =
            %Beaver.Changeset{name: "arith.addi", context: context, location: location}
            |> Beaver.Changeset.add_argument([input, MLIR.Operation.result(one, 0)])
            |> Beaver.Changeset.add_result(type)
            |> MLIR.Operation.create()

          MLIR.RewriterBase.set_insertion_point_before(base, operation)
          MLIR.RewriterBase.insert(base, one)
          MLIR.RewriterBase.insert(base, add)
          MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, add)
          :ok
        end,
        version: "1.0"
      )

    # First context run
    ctx1 = MLIR.Context.create()

    try do
      assert Beaver.Slang.load(ctx1, Beaver.MLIR.CounterPlanTest) |> MLIR.LogicalResult.success?()

      module1 =
        MLIR.Module.create!(
          ~S"""
          module {
            func.func @increment(%arg0: i32) -> i32 {
              %0 = "counter_plan_test.inc"(%arg0) : (i32) -> i32
              return %0 : i32
            }
          }
          """,
          ctx: ctx1
        )

      assert {:ok, ^module1, _diagnostics} = Plan.run(plan, module1)

      pass_manager1 = MLIR.CAPI.mlirPassManagerCreate(ctx1)

      MLIR.CAPI.mlirPassManagerAddOwnedPass(
        pass_manager1,
        MLIR.CAPI.mlirCreateConversionArithToLLVMConversionPass()
      )

      MLIR.CAPI.mlirPassManagerAddOwnedPass(
        pass_manager1,
        MLIR.CAPI.mlirCreateConversionConvertFuncToLLVMPass()
      )

      assert {:ok, _} = MLIR.PassManager.run(pass_manager1, module1)
      MLIR.PassManager.destroy(pass_manager1)

      rendered1 = MLIR.to_string(module1)
      refute rendered1 =~ "counter_plan_test.inc"
      assert rendered1 =~ "llvm.func"
      MLIR.Module.destroy(module1)
    after
      MLIR.Context.destroy(ctx1)
    end

    # Second context run using the EXACT same Plan instance
    ctx2 = MLIR.Context.create()

    try do
      assert Beaver.Slang.load(ctx2, Beaver.MLIR.CounterPlanTest) |> MLIR.LogicalResult.success?()

      module2 =
        MLIR.Module.create!(
          ~S"""
          module {
            func.func @increment(%arg0: i32) -> i32 {
              %0 = "counter_plan_test.inc"(%arg0) : (i32) -> i32
              return %0 : i32
            }
          }
          """,
          ctx: ctx2
        )

      assert converted2 = Plan.run!(plan, module2)
      rendered2 = MLIR.to_string(converted2)
      refute rendered2 =~ "counter_plan_test.inc"
      assert rendered2 =~ "arith.addi"
      MLIR.Module.destroy(module2)
    after
      MLIR.Context.destroy(ctx2)
    end
  end

  test "composes with a Native rewrite descriptor without another pattern language", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)
    owner = self()

    descriptor =
      ConversionPlanNativePatterns.erase_drop(
        init_state: owner,
        destruct: fn pid -> send(pid, :native_plan_pattern_destroyed) end
      )

    plan =
      Plan.new(mode: :full, timeout: 1_000)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_illegal_dialect("plan_native")
      |> Plan.add_pattern(descriptor, version: "1.0")

    module = MLIR.Module.create!(~s[module { "plan_native.drop"() : () -> () }], ctx: ctx)

    assert {:ok, ^module, []} = Plan.run(plan, module)
    assert_receive :native_plan_rewrite
    assert_receive :native_plan_pattern_destroyed
    refute MLIR.to_string(module) =~ "plan_native.drop"

    assert List.last(Plan.declaration(plan).entries) == %{
             kind: :add_pattern,
             name: :erase_drop,
             root: "plan_native.drop",
             benefit: 1,
             version: "1.0"
           }

    MLIR.Module.destroy(module)
  end

  test "full vs partial mode in Plan", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)

    partial_plan =
      Plan.new(mode: :partial, timeout: 1_000)
      |> Plan.add_legal_dialect("builtin")

    module = MLIR.Module.create!(~s[module { "foo.unknown"() : () -> () }], ctx: ctx)
    assert {:ok, ^module, []} = Plan.run(partial_plan, module)
    MLIR.Module.destroy(module)

    full_plan =
      Plan.new(mode: :full, timeout: 1_000)
      |> Plan.add_legal_dialect("builtin")

    full_module = MLIR.Module.create!(~s[module { "foo.unknown"() : () -> () }], ctx: ctx)

    assert {:error, %MLIR.Conversion.Error{mode: :full, diagnostics: diagnostics} = error} =
             Plan.run(full_plan, full_module)

    assert diagnostics != []
    assert Exception.message(error) =~ "foo.unknown"
    MLIR.Module.destroy(full_module)
  end

  test "source, target, and 1:N target materializations run through Plan", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)
    owner = self()
    i32 = MLIR.Type.i32(ctx: ctx)
    i64 = MLIR.Type.i64(ctx: ctx)

    source_module =
      MLIR.Module.create!(
        ~S[module { %a = arith.constant 1 : i64 %s = "foo.source"() : () -> i32 "foo.keep"(%s) : (i32) -> () }],
        ctx: ctx
      )

    [constant | _] =
      source_module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.to_list()

    replacement = MLIR.Operation.result(constant, 0)

    source_plan =
      Plan.new(mode: :full, timeout: 1_000, build_materializations: true)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_legal_dialect("arith")
      |> Plan.add_illegal_dialect("foo")
      |> Plan.add_legal_op("foo.keep")
      |> Plan.add_conversion(
        fn type -> if MLIR.equal?(type, i32), do: i64, else: :declined end,
        version: "1.0"
      )
      |> Plan.add_source_materialization(
        fn rewriter, output_type, inputs, loc ->
          send(owner, {:source_materialized, MLIR.to_string(output_type), length(inputs)})
          [value] = build_unrealized_cast(ctx, rewriter, [output_type], inputs, loc)
          value
        end,
        version: "1.0"
      )
      |> Plan.add_conversion_pattern(
        "foo.source",
        fn operation, [], rewriter ->
          MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, replacement)
          :ok
        end,
        version: "1.0"
      )

    assert {:ok, ^source_module, []} = Plan.run(source_plan, source_module)
    assert_receive {:source_materialized, "i32", 1}
    assert MLIR.to_string(source_module) =~ "unrealized_conversion_cast"
    MLIR.Module.destroy(source_module)

    target_module =
      MLIR.Module.create!(
        ~S[module { %a = "foo.producer"() : () -> i32 "foo.sink"(%a) : (i32) -> () }],
        ctx: ctx
      )

    target_plan =
      Plan.new(mode: :partial, timeout: 1_000, build_materializations: true)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_illegal_dialect("foo")
      |> Plan.add_legal_op("foo.producer")
      |> Plan.add_legal_op("foo.sink_legal")
      |> Plan.add_conversion(
        fn type ->
          send(owner, {:target_conversion, MLIR.to_string(type)})
          if MLIR.equal?(type, i32), do: i64, else: type
        end,
        version: "1.0"
      )
      |> Plan.add_target_materialization(
        fn rewriter, output_type, inputs, loc, original_type ->
          send(
            owner,
            {:target_materialized, MLIR.to_string(output_type), length(inputs),
             MLIR.to_string(original_type)}
          )

          [value] = build_unrealized_cast(ctx, rewriter, [output_type], inputs, loc)
          value
        end,
        version: "1.0"
      )
      |> Plan.add_conversion_pattern(
        "foo.sink",
        fn operation, [converted], rewriter ->
          send(owner, {:target_operand, MLIR.to_string(MLIR.Value.type(converted))})

          base = MLIR.ConversionPatternRewriter.as_base(rewriter)

          legal_sink =
            %Beaver.Changeset{
              name: "foo.sink_legal",
              context: MLIR.context(operation),
              location: MLIR.Operation.location(operation)
            }
            |> Beaver.Changeset.add_argument(converted)
            |> MLIR.Operation.create()

          MLIR.RewriterBase.insert(base, legal_sink)
          MLIR.ConversionPatternRewriter.erase_op(rewriter, operation)
          :ok
        end,
        version: "1.0"
      )

    assert {:ok, ^target_module, []} = Plan.run(target_plan, target_module)

    assert_receive {:target_conversion, "i32"}
    assert_receive {:target_operand, "i64"}
    assert_receive {:target_materialized, "i64", 1, "i32"}
    assert MLIR.to_string(target_module) =~ "unrealized_conversion_cast"
    MLIR.Module.destroy(target_module)

    one_to_n_module =
      MLIR.Module.create!(
        ~S[module { %a = arith.constant 1 : i32 "foo.sink"(%a) : (i32) -> () }],
        ctx: ctx
      )

    one_to_n_plan =
      Plan.new(mode: :full, timeout: 1_000, build_materializations: true)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_legal_dialect("arith")
      |> Plan.add_illegal_dialect("foo")
      |> Plan.add_1_to_n_conversion(
        fn type -> if MLIR.equal?(type, i32), do: [i64, i64], else: :declined end,
        version: "1.0"
      )
      |> Plan.add_1_to_n_target_materialization(
        fn rewriter, output_types, inputs, loc, _original ->
          send(
            owner,
            {:one_to_n_target_materialized, Enum.map(output_types, &MLIR.to_string/1),
             length(inputs)}
          )

          build_unrealized_cast(ctx, rewriter, output_types, inputs, loc)
        end,
        version: "1.0"
      )
      |> Plan.add_conversion_pattern(
        "foo.sink",
        fn operation, [[lhs, rhs]], rewriter ->
          context = MLIR.context(operation)
          base = MLIR.ConversionPatternRewriter.as_base(rewriter)

          add =
            %Beaver.Changeset{
              name: "arith.addi",
              context: context,
              location: MLIR.Operation.location(operation)
            }
            |> Beaver.Changeset.add_argument([lhs, rhs])
            |> Beaver.Changeset.add_result(i64)
            |> MLIR.Operation.create()

          MLIR.RewriterBase.insert(base, add)
          MLIR.ConversionPatternRewriter.erase_op(rewriter, operation)
          :ok
        end,
        one_to_n: true,
        version: "1.0"
      )

    assert {:ok, ^one_to_n_module, []} = Plan.run(one_to_n_plan, one_to_n_module)
    assert_receive {:one_to_n_target_materialized, ["i64", "i64"], 1}

    rendered = MLIR.to_string(one_to_n_module)
    assert rendered =~ "arith.addi"
    refute rendered =~ "foo.sink"
    MLIR.Module.destroy(one_to_n_module)
  end

  test "an erasing 1:N conversion supplies an empty operand range", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)
    owner = self()
    i32 = MLIR.Type.i32(ctx: ctx)

    plan =
      Plan.new(mode: :full, timeout: 1_000)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_legal_op("foo.producer")
      |> Plan.add_legal_op("foo.sink_legal")
      |> Plan.add_illegal_op("foo.sink")
      |> Plan.add_1_to_n_conversion(
        fn type -> if MLIR.equal?(type, i32), do: [], else: [type] end,
        version: "1.0"
      )
      |> Plan.add_conversion_pattern(
        "foo.sink",
        fn operation, [converted], rewriter ->
          send(owner, {:erased_operand_count, length(converted)})
          base = MLIR.ConversionPatternRewriter.as_base(rewriter)

          legal_sink =
            %Beaver.Changeset{
              name: "foo.sink_legal",
              context: MLIR.context(operation),
              location: MLIR.Operation.location(operation)
            }
            |> MLIR.Operation.create()

          MLIR.RewriterBase.insert(base, legal_sink)
          MLIR.ConversionPatternRewriter.erase_op(rewriter, operation)
          :ok
        end,
        one_to_n: true,
        version: "1.0"
      )

    module =
      MLIR.Module.create!(
        ~S[module { %a = "foo.producer"() : () -> i32 "foo.sink"(%a) : (i32) -> () }],
        ctx: ctx
      )

    assert {:ok, ^module, []} = Plan.run(plan, module)
    assert_receive {:erased_operand_count, 0}
    assert MLIR.to_string(module) =~ ~s["foo.sink_legal"()]
    refute MLIR.to_string(module) =~ ~s["foo.sink"(]
    MLIR.Module.destroy(module)
  end

  test "cleanup on success, conversion failure, callback exception, and callback timeout",
       %{
         ctx: ctx
       } do
    MLIR.Context.allow_unregistered_dialects(ctx)

    # 1. Conversion failure cleanup
    fail_plan =
      Plan.new(mode: :full, timeout: 1_000)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_illegal_dialect("foo")
      |> add_lifetime_probe()

    fail_module = MLIR.Module.create!(~s[module { "foo.illegal"() : () -> () }], ctx: ctx)
    assert {:error, %MLIR.Conversion.Error{mode: :full}} = Plan.run(fail_plan, fail_module)
    MLIR.Module.destroy(fail_module)

    # 2. Callback exception cleanup
    exc_plan =
      Plan.new(mode: :full, timeout: 1_000)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_dynamically_legal_op("foo.dynamic", fn _ ->
        raise "legality error in plan"
      end)
      |> add_lifetime_probe()

    exc_module = MLIR.Module.create!(~s[module { "foo.dynamic"() : () -> () }], ctx: ctx)

    assert {:error,
            %MLIR.Conversion.Error{
              callback_failure:
                {:exception, :error, %RuntimeError{message: "legality error in plan"}, _}
            }} = Plan.run(exc_plan, exc_module)

    MLIR.Module.destroy(exc_module)

    # 3. Callback timeout cleanup
    timeout_plan =
      Plan.new(mode: :full, timeout: 50)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_dynamically_legal_op("foo.slow", fn _ ->
        Process.sleep(200)
        :legal
      end)
      |> add_lifetime_probe()

    timeout_module = MLIR.Module.create!(~s[module { "foo.slow"() : () -> () }], ctx: ctx)
    assert {:error, %MLIR.Conversion.Error{}} = Plan.run(timeout_plan, timeout_module)
    MLIR.Module.destroy(timeout_module)
  end

  test "Plan.declaration/1 and compile/runtime option validation", %{ctx: ctx} do
    # Option validation in Plan.new
    assert_raise ArgumentError, ~r/unsupported Plan options/, fn ->
      Plan.new(invalid_opt: 123)
    end

    assert_raise ArgumentError, ~r/:mode must be :full or :partial/, fn ->
      Plan.new(mode: :invalid)
    end

    assert_raise ArgumentError, ~r/:timeout must be a non-negative integer/, fn ->
      Plan.new(timeout: -10)
    end

    assert_raise ArgumentError, ~r/unsupported conversion folding mode/, fn ->
      Plan.new(folding_mode: :invalid)
    end

    assert_raise ArgumentError, ~r/build_materializations must be boolean/, fn ->
      Plan.new(build_materializations: 123)
    end

    # Option validation in builder functions
    plan = Plan.new()

    assert_raise ArgumentError, ~r/unsupported add_conversion_pattern options/, fn ->
      Plan.add_conversion_pattern(plan, "foo.op", fn _, _, _ -> :ok end, invalid: true)
    end

    assert_raise ArgumentError, ~r/unsupported add_pattern options/, fn ->
      Plan.add_pattern(
        plan,
        %Beaver.Pattern.Native.Descriptor{
          name: :dummy,
          root: "arith.addi",
          match_and_rewrite: fn _, _, _, _ -> {:ok, nil} end
        },
        benefit: 2
      )
    end

    # Plan.declaration/1 structure assertion
    declared_plan =
      Plan.new(mode: :full, folding_mode: :before_patterns, timeout: 2_000)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_illegal_op("foo.bar")
      |> Plan.add_dynamically_legal_op("foo.dynamic", fn _ -> :legal end, version: "2.1")
      |> Plan.add_conversion(fn t -> t end)
      |> Plan.add_conversion_pattern("foo.pattern", fn _, _, _ -> :ok end,
        benefit: 2,
        version: "1.0"
      )

    decl = Plan.declaration(declared_plan)

    assert decl.mode == :full
    assert decl.folding_mode == :before_patterns
    assert decl.timeout == 2_000
    assert length(decl.entries) == 5

    assert Enum.at(decl.entries, 0) == %{kind: :add_legal_dialect, dialect: "builtin"}
    assert Enum.at(decl.entries, 1) == %{kind: :add_illegal_op, op: "foo.bar"}

    assert Enum.at(decl.entries, 2) == %{
             kind: :add_dynamically_legal_op,
             op: "foo.dynamic",
             version: "2.1"
           }

    assert Enum.at(decl.entries, 3) == %{kind: :add_conversion, version: :unversioned}

    assert Enum.at(decl.entries, 4) == %{
             kind: :add_conversion_pattern,
             root: "foo.pattern",
             benefit: 2,
             one_to_n: false,
             timeout: nil,
             version: "1.0"
           }

    nil_timeout_plan =
      Plan.new(timeout: nil)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_conversion_pattern(
        "plan.never",
        fn _operation, _operands, _rewriter -> :no_match end,
        timeout: nil,
        version: "1.0"
      )

    empty_module = MLIR.Module.create!("module {}", ctx: ctx)
    assert {:ok, ^empty_module, []} = Plan.run(nil_timeout_plan, empty_module)
    MLIR.Module.destroy(empty_module)
  end

  defp build_unrealized_cast(ctx, rewriter, output_types, inputs, loc) do
    changeset =
      %Beaver.Changeset{
        name: "builtin.unrealized_conversion_cast",
        context: ctx,
        location: loc
      }
      |> Beaver.Changeset.add_argument(inputs)

    operation =
      Enum.reduce(output_types, changeset, &Beaver.Changeset.add_result(&2, &1))
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(rewriter, operation)
    operation |> MLIR.Operation.results() |> Enum.to_list()
  end

  defp add_lifetime_probe(plan) do
    Plan.add_conversion_pattern(
      plan,
      "plan.never",
      fn _operation, _operands, _rewriter -> :no_match end,
      version: "lifetime-probe"
    )
  end
end
