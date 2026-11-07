defmodule Beaver.MLIR.ConversionTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  import Beaver.ConversionHelpers

  test "lowers a Slang dialect with per-instance legality and a 1:N type conversion", %{
    ctx: ctx
  } do
    assert ConversionSlang
           |> then(&Beaver.Slang.load(ctx, &1))
           |> MLIR.LogicalResult.success?()

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          %a = arith.constant 1 : i64
          %b = arith.constant 2 : i64
          %source = "conversion_test.source"() : () -> i32
          "conversion_test.sink"(%source) : (i32) -> ()
          "conversion_test.keep"() : () -> ()
        }
        """,
        ctx: ctx
      )

    [first, second | _] =
      module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.to_list()

    replacements = [MLIR.Operation.result(first, 0), MLIR.Operation.result(second, 0)]
    owner = self()

    target =
      MLIR.ConversionTarget.create(ctx, timeout: 1_000)
      |> MLIR.ConversionTarget.add_legal_dialect("builtin")
      |> MLIR.ConversionTarget.add_legal_dialect("arith")
      |> MLIR.ConversionTarget.add_illegal_dialect("conversion_test")
      |> MLIR.ConversionTarget.add_dynamically_legal_op(
        "conversion_test.keep",
        fn operation ->
          send(owner, {:checked_legality, MLIR.Operation.name(operation)})
          :legal
        end
      )

    i32 = MLIR.Type.i32(ctx: ctx)
    i64 = MLIR.Type.i64(ctx: ctx)

    converter =
      MLIR.TypeConverter.create(
        one_to_n: fn type ->
          if MLIR.equal?(type, i32), do: [i64, i64], else: :declined
        end,
        timeout: 1_000
      )

    patterns = MLIR.RewritePatternSet.create(ctx)

    MLIR.ConversionPattern.add(
      patterns,
      "conversion_test.source",
      converter,
      fn operation, [], rewriter ->
        MLIR.ConversionPatternRewriter.replace_op_with_multiple(
          rewriter,
          operation,
          [replacements]
        )

        :ok
      end,
      ctx: ctx,
      one_to_n: true,
      timeout: 1_000
    )

    MLIR.ConversionPattern.add(
      patterns,
      "conversion_test.sink",
      converter,
      fn operation, [converted_operands], rewriter ->
        send(owner, {:converted_operands, converted_operands})
        MLIR.ConversionPatternRewriter.erase_op(rewriter, operation)
        :ok
      end,
      ctx: ctx,
      one_to_n: true,
      timeout: 1_000
    )

    try do
      assert {:ok, ^module, []} = MLIR.Conversion.full(module, target, patterns, timeout: 1_000)
      assert_receive {:checked_legality, "conversion_test.keep"}
      assert_receive {:converted_operands, converted_operands}
      assert length(converted_operands) == 2

      rendered = MLIR.to_string(module)
      refute rendered =~ "conversion_test.source"
      refute rendered =~ "conversion_test.sink"
      assert rendered =~ "conversion_test.keep"
    after
      assert :ok = MLIR.TypeConverter.destroy(converter)
      assert :ok = MLIR.ConversionTarget.destroy(target)
      MLIR.Module.destroy(module)
    end
  end

  test "partial conversion permits unknown operations while full conversion returns diagnostics",
       %{ctx: ctx} do
    for mode <- [:partial, :full] do
      module = unknown_module(ctx)
      target = legal_builtin_target(ctx)
      patterns = MLIR.RewritePatternSet.create(ctx)

      try do
        result = apply(MLIR.Conversion, mode, [module, target, patterns, [timeout: 1_000]])

        if mode == :partial do
          assert {:ok, ^module, []} = result
        else
          assert {:error, %MLIR.Conversion.Error{mode: :full, diagnostics: diagnostics} = error} =
                   result

          assert diagnostics != []
          assert Exception.message(error) =~ "foo.unknown"
        end
      after
        assert :ok = MLIR.ConversionTarget.destroy(target)
        MLIR.Module.destroy(module)
      end
    end
  end

  test "profiling returns bounded native and BEAM callback aggregates", %{ctx: ctx} do
    module = unknown_module(ctx, "foo.profiled")

    target =
      legal_builtin_target(ctx)
      |> MLIR.ConversionTarget.add_dynamically_legal_op("foo.profiled", fn _ -> :legal end)

    patterns = MLIR.RewritePatternSet.create(ctx)

    assert {{:ok, ^module, []}, receipt} =
             MLIR.Conversion.profile(:full, module, target, patterns, timeout: 1_000)

    assert receipt["schema_version"] == 2
    assert receipt["status"] == "ok"
    assert receipt["duration_ns"] >= receipt["native"]["duration_ns"]
    assert receipt["native"]["duration_ns"] > 0

    assert receipt["native"]["callback_wait_sum_ns"] >=
             receipt["native"]["callback_wait_max_ns"]

    assert receipt["native"]["target_lock_wait_ns"] <= receipt["native"]["duration_ns"]
    assert receipt["native"]["unattributed_residual_ns"] <= receipt["native"]["duration_ns"]
    assert receipt["process_cpu_time_ns"] >= 0
    assert receipt["rss"]["peak_after_bytes"] >= receipt["rss"]["peak_before_bytes"]
    assert receipt["ir"]["before"] == receipt["ir"]["after"]
    assert receipt["ir"]["before"]["operations"] == 2
    assert receipt["beam"]["reductions"] > 0
    assert receipt["beam"]["peak_process_memory_bytes"] > 0
    assert receipt["beam"]["callback_count"] > 0
    assert receipt["beam"]["max_in_flight"] == 1
    assert receipt["beam"]["memory_sample_interval_callbacks"] == 256
    assert length(receipt["hotspots"]) in 1..3

    assert receipt["hotspots"] ==
             Enum.sort_by(
               receipt["hotspots"],
               &{-&1["duration_ns"], &1["source"], &1["callback_kind"] || "",
                &1["callback_root"] || ""}
             )

    legality = Enum.find(receipt["callbacks"], &(&1["kind"] == "conversion_legality"))
    assert legality["count"] > 0
    assert legality["native_wait_sum_ns"] >= legality["beam_service_ns"]
    assert legality["max_native_wait_ns"] >= legality["max_beam_service_ns"]

    assert :ok = MLIR.ConversionTarget.destroy(target)
    MLIR.Module.destroy(module)
  end

  test "profile aggregates preserve totals and maxima across roots", %{ctx: ctx} do
    alias Beaver.MLIR.Conversion.Profile
    module = unknown_module(ctx, "foo.profiled")

    state =
      Enum.reduce(1..257, Profile.start(:full, module), fn i, acc ->
        Profile.record_callback(acc, :conversion_pattern, "ex.func", i, i + 7)
      end)

    state = Profile.record_callback(state, :conversion_pattern, "ex.return", 4, 9)
    receipt = Profile.finish(state, {:ok, module, []}, module, {100_000, 0})
    assert receipt["beam"]["callback_count"] == 258
    [func, return] = receipt["callbacks"]
    keys = ~w(count beam_service_ns max_beam_service_ns native_wait_sum_ns max_native_wait_ns)
    assert func["root"] == "ex.func"
    assert Enum.map(keys, &func[&1]) == [257, 33_153, 257, 34_952, 264]
    assert return["root"] == "ex.return"
    assert Enum.map(keys, &return[&1]) == [1, 4, 4, 9, 9]
    MLIR.Module.destroy(module)
  end

  test "type conversion supports locked and unlocked contexts and closed handles", %{ctx: ctx} do
    unlocked = MLIR.Context.create(threading: false, thread_safe: false)
    on_exit(fn -> MLIR.Context.destroy(unlocked) end)

    for ctx <- [ctx, unlocked] do
      i32 = MLIR.Type.i32(ctx: ctx)
      i64 = MLIR.Type.i64(ctx: ctx)
      f32 = MLIR.Type.f32(ctx: ctx)

      converter =
        MLIR.TypeConverter.create()
        |> MLIR.TypeConverter.add_conversion_map([i32], i64)

      assert {:ok, converted} = MLIR.TypeConverter.convert(converter, i32)
      assert MLIR.equal?(converted, i64)
      assert {:ok, unchanged} = MLIR.TypeConverter.convert(converter, f32)
      assert MLIR.equal?(unchanged, f32)
      assert :ok = MLIR.TypeConverter.destroy(converter)
      assert {:error, :conversion_failed} = MLIR.TypeConverter.convert(converter, i32)

      MLIR.TypeConverter.with([conversion: fn type -> type end], fn callback ->
        assert {:ok, converted} = MLIR.TypeConverter.convert(callback, i32)
        assert MLIR.equal?(converted, i32)
      end)
    end
  end

  test "callback failures are attributed and a dead callback owner fails deterministically", %{
    ctx: ctx
  } do
    module = unknown_module(ctx, "foo.dynamic")

    target =
      legal_builtin_target(ctx)
      |> MLIR.ConversionTarget.add_dynamically_legal_op("foo.dynamic", fn _ ->
        raise "legality exploded"
      end)

    patterns = MLIR.RewritePatternSet.create(ctx)

    assert {:error,
            %MLIR.Conversion.Error{
              callback_failure: {:exception, :error, %RuntimeError{}, _stacktrace}
            }} = MLIR.Conversion.full(module, target, patterns, timeout: 1_000)

    assert :ok = MLIR.ConversionTarget.destroy(target)
    MLIR.Module.destroy(module)

    parent = self()

    {owner, monitor} =
      spawn_monitor(fn ->
        owned_target =
          legal_builtin_target(ctx)
          |> MLIR.ConversionTarget.add_dynamically_legal_op("foo.dynamic", fn _ -> :legal end)

        send(parent, {:owned_target, owned_target})
        Process.sleep(:infinity)
      end)

    assert_receive {:owned_target, owned_target}
    Process.exit(owner, :kill)
    assert_receive {:DOWN, ^monitor, :process, ^owner, :killed}

    ownerless_module = unknown_module(ctx, "foo.dynamic")
    ownerless_patterns = MLIR.RewritePatternSet.create(ctx)

    assert {:error, %MLIR.Conversion.Error{}} =
             MLIR.Conversion.full(
               ownerless_module,
               owned_target,
               ownerless_patterns,
               timeout: 100
             )

    assert :ok = MLIR.ConversionTarget.destroy(owned_target)
    assert :ok = MLIR.ConversionTarget.destroy(owned_target)
    MLIR.Module.destroy(ownerless_module)
  end

  test "source and 1:N target materializations run through Elixir callbacks", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)
    owner = self()
    i32 = MLIR.Type.i32(ctx: ctx)
    i64 = MLIR.Type.i64(ctx: ctx)

    source_module = materialization_module(ctx, :source)

    [constant | _] =
      source_module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.to_list()

    replacement = MLIR.Operation.result(constant, 0)

    source_target =
      legal_arith_target(ctx)
      |> MLIR.ConversionTarget.add_illegal_dialect("foo")
      |> MLIR.ConversionTarget.add_legal_op("foo.keep")

    source_converter =
      MLIR.TypeConverter.create(
        conversion: fn type -> if MLIR.equal?(type, i32), do: i64, else: :declined end,
        source_materialization: fn rewriter, output_type, inputs, loc ->
          send(owner, {:source_materialized, output_type, inputs})
          [value] = build_unrealized_cast(rewriter, [output_type], inputs, loc)
          value
        end,
        timeout: 1_000
      )

    source_patterns = MLIR.RewritePatternSet.create(ctx)

    MLIR.ConversionPattern.add(
      source_patterns,
      "foo.source",
      source_converter,
      fn operation, [], rewriter ->
        MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, replacement)
        :ok
      end,
      ctx: ctx,
      timeout: 1_000
    )

    assert {:ok, ^source_module, []} =
             MLIR.Conversion.full(
               source_module,
               source_target,
               source_patterns,
               timeout: 1_000,
               build_materializations: true
             )

    assert_receive {:source_materialized, materialized_source_type, [_input]}
    assert MLIR.equal?(materialized_source_type, i32)
    assert MLIR.to_string(source_module) =~ "unrealized_conversion_cast"
    assert :ok = MLIR.TypeConverter.destroy(source_converter)
    assert :ok = MLIR.ConversionTarget.destroy(source_target)
    MLIR.Module.destroy(source_module)

    target_module = materialization_module(ctx, :one_to_n)

    target = legal_arith_target(ctx) |> MLIR.ConversionTarget.add_illegal_dialect("foo")

    target_converter =
      MLIR.TypeConverter.create(
        one_to_n: fn type -> if MLIR.equal?(type, i32), do: [i64, i64], else: :declined end,
        one_to_n_target_materialization: fn rewriter, output_types, inputs, loc, _original ->
          send(owner, {:target_materialized, output_types, inputs})
          build_unrealized_cast(rewriter, output_types, inputs, loc)
        end,
        timeout: 1_000
      )

    target_patterns = MLIR.RewritePatternSet.create(ctx)

    MLIR.ConversionPattern.add(
      target_patterns,
      "foo.sink",
      target_converter,
      fn operation, [[lhs, rhs]], rewriter ->
        base = MLIR.ConversionPatternRewriter.as_base(rewriter)

        insert(base, "arith.addi", MLIR.Operation.location(operation), [lhs, rhs], i64)
        MLIR.ConversionPatternRewriter.erase_op(rewriter, operation)
        :ok
      end,
      ctx: ctx,
      one_to_n: true,
      timeout: 1_000
    )

    assert {:ok, ^target_module, []} =
             MLIR.Conversion.full(
               target_module,
               target,
               target_patterns,
               timeout: 1_000,
               build_materializations: true
             )

    assert_receive {:target_materialized, materialized_target_types, [_input]}
    assert Enum.all?(materialized_target_types, &MLIR.equal?(&1, i64))
    rendered = MLIR.to_string(target_module)
    assert rendered =~ "unrealized_conversion_cast"
    assert rendered =~ "arith.addi"
    assert :ok = MLIR.TypeConverter.destroy(target_converter)
    assert :ok = MLIR.ConversionTarget.destroy(target)
    MLIR.Module.destroy(target_module)
  end

  test "a pattern set keeps its type converter alive until pattern destruction", %{ctx: ctx} do
    converter = MLIR.TypeConverter.create(conversion: fn type -> type end)
    patterns = MLIR.RewritePatternSet.create(ctx)

    MLIR.ConversionPattern.add(
      patterns,
      "arith.constant",
      converter,
      fn _operation, _operands, _rewriter -> :no_match end,
      ctx: ctx
    )

    assert_raise Kinda.CallError, fn -> MLIR.TypeConverter.destroy(converter) end
    assert :ok = MLIR.RewritePatternSet.threaded_destroy(ctx, patterns)
    assert :ok = MLIR.TypeConverter.destroy(converter)
    assert :ok = MLIR.TypeConverter.destroy(converter)
  end

  test "conversion-owned patterns and config are cleaned when the caller terminates", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)
    parent = self()

    {owner, monitor} =
      spawn_monitor(fn ->
        module = unknown_module(ctx, "foo.dynamic")

        target =
          legal_builtin_target(ctx)
          |> MLIR.ConversionTarget.add_dynamically_legal_op("foo.dynamic", fn _operation ->
            send(parent, :owner_callback_started)
            Process.sleep(:infinity)
          end)

        converter = MLIR.TypeConverter.create(conversion: fn type -> type end, timeout: 100)
        patterns = MLIR.RewritePatternSet.create(ctx)

        MLIR.ConversionPattern.add(
          patterns,
          "arith.constant",
          converter,
          fn _operation, _operands, _rewriter -> :no_match end,
          ctx: ctx,
          timeout: 100
        )

        send(parent, {:conversion_owner_resources, module, target, converter})

        receive do
          :start_conversion -> MLIR.Conversion.full(module, target, patterns, timeout: 100)
        end
      end)

    try do
      assert_receive {:conversion_owner_resources, module, target, converter}, 5_000

      try do
        send(owner, :start_conversion)
        assert_receive :owner_callback_started, 5_000
        Process.exit(owner, :kill)
        assert_receive {:DOWN, ^monitor, :process, ^owner, :killed}, 5_000
      after
        ensure_process_terminated(owner)
        assert :ok = eventually_destroy_converter(converter, 500)
        assert :ok = MLIR.ConversionTarget.destroy(target)
        MLIR.Module.destroy(module)
      end
    after
      ensure_process_terminated(owner)
    end
  end

  defp legal_builtin_target(ctx) do
    MLIR.ConversionTarget.create(ctx, timeout: 1_000)
    |> MLIR.ConversionTarget.add_legal_dialect("builtin")
  end

  defp legal_arith_target(ctx) do
    legal_builtin_target(ctx)
    |> MLIR.ConversionTarget.add_legal_dialect("arith")
  end

  defp eventually_destroy_converter(converter, attempts) do
    MLIR.TypeConverter.destroy(converter)
  rescue
    exception in Kinda.CallError ->
      if attempts == 0 do
        reraise exception, __STACKTRACE__
      else
        receive do
        after
          10 -> eventually_destroy_converter(converter, attempts - 1)
        end
      end
  end

  defp ensure_process_terminated(process) do
    monitor = Process.monitor(process)

    if Process.alive?(process) do
      Process.exit(process, :kill)
    end

    receive do
      {:DOWN, ^monitor, :process, ^process, _reason} -> :ok
    after
      5_000 -> raise "timed out terminating conversion owner"
    end
  end

  defp unknown_module(ctx, name \\ "foo.unknown") do
    MLIR.Context.allow_unregistered_dialects(ctx)

    mlir ctx: ctx do
      module do
        ~o/#{name}/ >>> []
      end
    end
    |> MLIR.verify!()
  end
end
