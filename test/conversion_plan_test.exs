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
  import Beaver.ConversionHelpers

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
          base = MLIR.ConversionPatternRewriter.as_base(rewriter)
          type = MLIR.Value.type(input)
          location = MLIR.Operation.location(operation)
          MLIR.RewriterBase.set_insertion_point_before(base, operation)

          one =
            insert(
              base,
              "arith.constant",
              location,
              [value: MLIR.Attribute.integer(type, 1)],
              type
            )

          add = insert(base, "arith.addi", location, [input, MLIR.Operation.result(one, 0)], type)
          MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, add)
          :ok
        end,
        version: "1.0"
      )

    for driver <- [:run, :run!] do
      Beaver.Native.with_resource(MLIR.Context.create(), &MLIR.Context.destroy/1, fn ctx ->
        assert Beaver.Slang.load(ctx, Beaver.MLIR.CounterPlanTest)
               |> MLIR.LogicalResult.success?()

        module =
          MLIR.Module.create!(
            ~S"""
            module {
              func.func @increment(%arg0: i32) -> i32 {
                %0 = "counter_plan_test.inc"(%arg0) : (i32) -> i32
                return %0 : i32
              }
            }
            """,
            ctx: ctx
          )

        Beaver.Native.with_resource(module, &MLIR.Module.destroy/1, fn module ->
          if driver == :run do
            assert {:ok, ^module, _diagnostics} = Plan.run(plan, module)

            Beaver.Composer.append(module, "convert-arith-to-llvm,convert-func-to-llvm")
            |> Beaver.Composer.run!()

            assert MLIR.to_string(module) =~ "llvm.func"
          else
            assert ^module = Plan.run!(plan, module)
            assert MLIR.to_string(module) =~ "arith.addi"
          end

          refute MLIR.to_string(module) =~ "counter_plan_test.inc"
        end)
      end)
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

  test "materializes a pattern population callback once per run", %{ctx: ctx} do
    owner = self()

    plan =
      Plan.new(mode: :full)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_pattern_population(
        fn patterns, converter ->
          send(owner, {:populated, patterns, converter})
          :ok
        end,
        version: "1.0"
      )

    module = MLIR.Module.create!("module {}", ctx: ctx)
    assert {:ok, ^module, []} = Plan.run(plan, module)

    assert_receive {:populated, %MLIR.RewritePatternSet{}, %MLIR.TypeConverter{}}

    assert List.last(Plan.declaration(plan).entries) == %{
             kind: :add_pattern_population,
             version: "1.0"
           }

    MLIR.Module.destroy(module)
  end

  test "pattern population fails closed on an invalid outcome", %{ctx: ctx} do
    plan =
      Plan.new(mode: :full)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_pattern_population(fn _patterns, _converter -> :ignored end)

    module = MLIR.Module.create!("module {}", ctx: ctx)

    assert_raise ArgumentError, ~r/pattern population must return :ok/, fn ->
      Plan.run(plan, module)
    end

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

    source_module = materialization_module(ctx, :source)

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
          [value] = build_unrealized_cast(rewriter, [output_type], inputs, loc)
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

    target_module = materialization_module(ctx, :target)

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

          [value] = build_unrealized_cast(rewriter, [output_type], inputs, loc)
          value
        end,
        version: "1.0"
      )
      |> Plan.add_conversion_pattern(
        "foo.sink",
        fn operation, [converted], rewriter ->
          send(owner, {:target_operand, MLIR.to_string(MLIR.Value.type(converted))})

          base = MLIR.ConversionPatternRewriter.as_base(rewriter)

          insert(base, "foo.sink_legal", MLIR.Operation.location(operation), converted)
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

    one_to_n_module = materialization_module(ctx, :one_to_n)

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

          build_unrealized_cast(rewriter, output_types, inputs, loc)
        end,
        version: "1.0"
      )
      |> Plan.add_conversion_pattern(
        "foo.sink",
        fn operation, [[lhs, rhs]], rewriter ->
          base = MLIR.ConversionPatternRewriter.as_base(rewriter)
          insert(base, "arith.addi", MLIR.Operation.location(operation), [lhs, rhs], i64)
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

          insert(base, "foo.sink_legal", MLIR.Operation.location(operation), [])
          MLIR.ConversionPatternRewriter.erase_op(rewriter, operation)
          :ok
        end,
        one_to_n: true,
        version: "1.0"
      )

    module = materialization_module(ctx, :target)

    assert {:ok, ^module, []} = Plan.run(plan, module)
    assert_receive {:erased_operand_count, 0}
    assert MLIR.to_string(module) =~ ~s["foo.sink_legal"()]
    refute MLIR.to_string(module) =~ ~s["foo.sink"(]
    MLIR.Module.destroy(module)
  end

  test "cleanup on conversion failure, callback exception, and callback timeout", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)

    for scenario <- [:illegal, :exception, :timeout] do
      plan =
        Plan.new(mode: :full, timeout: if(scenario == :timeout, do: 50, else: 1_000))
        |> Plan.add_legal_dialect("builtin")
        |> add_lifetime_probe()

      plan =
        if scenario == :illegal do
          Plan.add_illegal_dialect(plan, "foo")
        else
          Plan.add_dynamically_legal_op(plan, "foo.dynamic", fn _ ->
            if scenario == :exception, do: raise("legality error in plan")
            Process.sleep(200)
            :legal
          end)
        end

      module = MLIR.Module.create!(~s[module { "foo.dynamic"() : () -> () }], ctx: ctx)

      Beaver.Native.with_resource(module, &MLIR.Module.destroy/1, fn module ->
        assert {:error, %MLIR.Conversion.Error{mode: :full} = error} = Plan.run(plan, module)

        if scenario == :exception do
          assert {:exception, :error, %RuntimeError{message: "legality error in plan"}, _} =
                   error.callback_failure
        end
      end)
    end
  end

  test "Plan.declaration/1 and compile/runtime option validation", %{ctx: ctx} do
    for {options, message} <- [
          {[invalid_opt: 123], ~r/unsupported Plan options/},
          {[mode: :invalid], ~r/:mode must be :full, :partial or :analysis/},
          {[timeout: -10], ~r/:timeout must be a non-negative integer/},
          {[folding_mode: :invalid], ~r/unsupported conversion folding mode/},
          {[build_materializations: 123], ~r/build_materializations must be boolean/}
        ] do
      assert_raise ArgumentError, message, fn -> Plan.new(options) end
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

    assert_raise ArgumentError, ~r/unsupported add_pattern_population options/, fn ->
      Plan.add_pattern_population(plan, fn _, _ -> :ok end, invalid: true)
    end

    # Plan.declaration/1 structure assertion
    declared_plan =
      Plan.new(mode: :full, folding_mode: :before_patterns, timeout: 2_000)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_illegal_op("foo.bar")
      |> Plan.add_dynamically_legal_op("foo.dynamic", fn _ -> :legal end, version: "2.1")
      |> Plan.add_conversion(fn t -> t end)
      |> Plan.add_conversion_map(["i32", "index"], "i64")
      |> Plan.add_conversion_pattern("foo.pattern", fn _, _, _ -> :ok end,
        benefit: 2,
        version: "1.0"
      )

    assert Plan.declaration(declared_plan) == %{
             mode: :full,
             folding_mode: :before_patterns,
             timeout: 2_000,
             build_materializations: nil,
             entries: [
               %{kind: :add_legal_dialect, dialect: "builtin"},
               %{kind: :add_illegal_op, op: "foo.bar"},
               %{kind: :add_dynamically_legal_op, op: "foo.dynamic", version: "2.1"},
               %{kind: :add_conversion, version: :unversioned},
               %{kind: :add_conversion_map, source_types: ["i32", "index"], target_type: "i64"},
               %{
                 kind: :add_conversion_pattern,
                 root: "foo.pattern",
                 benefit: 2,
                 one_to_n: false,
                 timeout: nil,
                 version: "1.0"
               }
             ]
           }

    nil_timeout_plan =
      Plan.new(timeout: nil)
      |> Plan.add_legal_dialect("builtin")
      |> add_lifetime_probe(timeout: nil, version: "1.0")

    empty_module = MLIR.Module.create!("module {}", ctx: ctx)
    assert {:ok, ^empty_module, []} = Plan.run(nil_timeout_plan, empty_module)
    MLIR.Module.destroy(empty_module)
  end

  defp add_lifetime_probe(plan, opts \\ []) do
    Plan.add_conversion_pattern(
      plan,
      "plan.never",
      fn _operation, _operands, _rewriter -> :no_match end,
      Keyword.put_new(opts, :version, "lifetime-probe")
    )
  end
end
