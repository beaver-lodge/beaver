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

  alias Beaver.MLIR.{
    Attribute,
    Context,
    Conversion,
    ConversionPatternRewriter,
    LogicalResult,
    Operation,
    RewriterBase,
    Type,
    Value
  }

  alias Beaver.MLIR
  alias Conversion.Plan
  alias Beaver.MLIR.ConversionPlanNativePatterns
  import Beaver.ConversionHelpers

  test "lowers a Slang dialect using Plan and reuses the Plan across fresh MLIR contexts" do
    plan =
      Plan.new(mode: :full, folding_mode: :after_patterns, build_materializations: true, timeout: 5000)
      |> Plan.add_legal_dialects(~w(builtin func arith))
      |> Plan.add_illegal_dialect("counter_plan_test")
      |> Plan.add_conversion(fn type -> type end, version: "1.0")
      |> Plan.add_conversion_pattern(
        "counter_plan_test.inc",
        fn operation, [input], rewriter ->
          base = ConversionPatternRewriter.as_base(rewriter)
          type = Value.type(input)
          location = Operation.location(operation)
          RewriterBase.set_insertion_point_before(base, operation)

          one = insert(base, "arith.constant", location, [value: Attribute.integer(type, 1)], type)

          add = insert(base, "arith.addi", location, [input, Operation.result(one, 0)], type)
          ConversionPatternRewriter.replace_op(rewriter, operation, add)
          :ok
        end,
        version: "1.0"
      )

    for driver <- [:run, :run!] do
      Beaver.Native.with_resource(Context.create(), &Context.destroy/1, fn ctx ->
        assert Beaver.Slang.load(ctx, Beaver.MLIR.CounterPlanTest) |> LogicalResult.success?()

        module =
          "module {\n  func.func @increment(%arg0: i32) -> i32 {\n    %0 = \"counter_plan_test.inc\"(%arg0) : (i32) -> i32\n    return %0 : i32\n  }\n}\n"
          |> MLIR.Module.create!(ctx: ctx)

        Beaver.MLIR.Module.with_module(module, fn module ->
          if driver == :run do
            assert {:ok, ^module, _diagnostics} = Plan.run(plan, module)

            Beaver.Composer.append(module, "convert-arith-to-llvm,convert-func-to-llvm") |> Beaver.Composer.run!()

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
    Context.allow_unregistered_dialects(ctx)
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

    assert List.last(Plan.declaration(plan).entries) == %{kind: :add_pattern_population, version: "1.0"}

    MLIR.Module.destroy(module)
  end

  test "pattern population fails closed on an invalid outcome", %{ctx: ctx} do
    plan =
      Plan.new(mode: :full)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_pattern_population(fn _patterns, _converter -> :ignored end)

    module = MLIR.Module.create!("module {}", ctx: ctx)

    assert_raise ArgumentError, ~r/pattern population must return :ok/, fn -> Plan.run(plan, module) end

    MLIR.Module.destroy(module)
  end

  test "full vs partial mode in Plan", %{ctx: ctx} do
    Context.allow_unregistered_dialects(ctx)

    partial_plan = Plan.new(mode: :partial, timeout: 1_000) |> Plan.add_legal_dialect("builtin")

    module = MLIR.Module.create!(~s[module { "foo.unknown"() : () -> () }], ctx: ctx)
    assert {:ok, ^module, []} = Plan.run(partial_plan, module)
    MLIR.Module.destroy(module)

    full_plan = Plan.new(mode: :full, timeout: 1_000) |> Plan.add_legal_dialect("builtin")

    full_module = MLIR.Module.create!(~s[module { "foo.unknown"() : () -> () }], ctx: ctx)

    assert {:error, %Conversion.Error{mode: :full, diagnostics: diagnostics} = error} = Plan.run(full_plan, full_module)

    assert diagnostics != []
    assert Exception.message(error) =~ "foo.unknown"
    MLIR.Module.destroy(full_module)
  end

  @materialization_callbacks %{
    conversion: :add_conversion,
    one_to_n: :add_1_to_n_conversion,
    source_materialization: :add_source_materialization,
    target_materialization: :add_target_materialization,
    one_to_n_target_materialization: :add_1_to_n_target_materialization
  }
  for kind <- [:source, :target, :one_to_n] do
    @tag materialization: kind
    test "#{kind} materialization runs through Plan", %{ctx: ctx, materialization: kind} do
      Context.allow_unregistered_dialects(ctx)
      fixture = materialization_fixture(ctx, kind)
      mode = if kind == :target, do: :partial, else: :full

      plan =
        Plan.new(mode: mode, timeout: 1_000, build_materializations: true)
        |> Plan.add_legal_dialects(fixture.dialects)
        |> Plan.add_illegal_dialect("foo")
        |> Plan.add_legal_ops(fixture.legal_ops)

      plan =
        Enum.reduce(fixture.callbacks, plan, fn {key, callback}, plan ->
          apply(Plan, Map.fetch!(@materialization_callbacks, key), [plan, callback, [version: "1.0"]])
        end)

      plan = Plan.add_conversion_pattern(plan, fixture.root, fixture.rewrite, [version: "1.0"] ++ fixture.pattern_opts)

      MLIR.Module.with_module(fixture.module, fn module ->
        assert {:ok, ^module, []} = Plan.run(plan, module)
        assert_materialized(module, kind)
      end)
    end
  end

  test "an erasing 1:N conversion supplies an empty operand range", %{ctx: ctx} do
    Context.allow_unregistered_dialects(ctx)
    owner = self()
    i32 = Type.i32(ctx: ctx)

    plan =
      Plan.new(mode: :full, timeout: 1_000)
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_legal_ops(~w(foo.producer foo.sink_legal))
      |> Plan.add_illegal_op("foo.sink")
      |> Plan.add_1_to_n_conversion(fn type -> if MLIR.equal?(type, i32), do: [], else: [type] end, version: "1.0")
      |> Plan.add_conversion_pattern(
        "foo.sink",
        fn operation, [converted], rewriter ->
          send(owner, {:erased_operand_count, length(converted)})
          base = ConversionPatternRewriter.as_base(rewriter)
          insert(base, "foo.sink_legal", Operation.location(operation), [])
          ConversionPatternRewriter.erase_op(rewriter, operation)
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
    Context.allow_unregistered_dialects(ctx)

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

      Beaver.MLIR.Module.with_module(module, fn module ->
        assert {:error, %Conversion.Error{mode: :full} = error} = Plan.run(plan, module)

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
      |> Plan.add_conversion_pattern("foo.pattern", fn _, _, _ -> :ok end, benefit: 2, version: "1.0")

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
      Plan.new(timeout: nil) |> Plan.add_legal_dialect("builtin") |> add_lifetime_probe(timeout: nil, version: "1.0")

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
