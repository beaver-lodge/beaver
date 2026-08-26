defmodule Beaver.MLIR.ConversionTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR

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
       %{
         ctx: ctx
       } do
    partial_module = unknown_module(ctx)
    partial_target = legal_builtin_target(ctx)
    partial_patterns = MLIR.RewritePatternSet.create(ctx)

    assert {:ok, ^partial_module, []} =
             MLIR.Conversion.partial(
               partial_module,
               partial_target,
               partial_patterns,
               timeout: 1_000
             )

    assert :ok = MLIR.ConversionTarget.destroy(partial_target)
    MLIR.Module.destroy(partial_module)

    full_module = unknown_module(ctx)
    full_target = legal_builtin_target(ctx)
    full_patterns = MLIR.RewritePatternSet.create(ctx)

    assert {:error, %MLIR.Conversion.Error{mode: :full, diagnostics: diagnostics} = error} =
             MLIR.Conversion.full(full_module, full_target, full_patterns, timeout: 1_000)

    assert diagnostics != []
    assert Exception.message(error) =~ "foo.unknown"
    assert :ok = MLIR.ConversionTarget.destroy(full_target)
    MLIR.Module.destroy(full_module)
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

    source_module =
      MLIR.Module.create!(
        ~S[module { %a = arith.constant 1 : i64 %s = "foo.source"() : () -> i32 "foo.keep"(%s) : (i32) -> () }],
        ctx: ctx
      )

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
          [value] = build_unrealized_cast(ctx, rewriter, [output_type], inputs, loc)
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

    target_module =
      MLIR.Module.create!(
        ~S[module { %a = arith.constant 1 : i32 "foo.sink"(%a) : (i32) -> () }],
        ctx: ctx
      )

    target = legal_arith_target(ctx) |> MLIR.ConversionTarget.add_illegal_dialect("foo")

    target_converter =
      MLIR.TypeConverter.create(
        one_to_n: fn type -> if MLIR.equal?(type, i32), do: [i64, i64], else: :declined end,
        one_to_n_target_materialization: fn rewriter, output_types, inputs, loc, _original ->
          send(owner, {:target_materialized, output_types, inputs})
          build_unrealized_cast(ctx, rewriter, output_types, inputs, loc)
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

        add =
          %Beaver.Changeset{
            name: "arith.addi",
            context: ctx,
            location: MLIR.Operation.location(operation)
          }
          |> Beaver.Changeset.add_argument([lhs, rhs])
          |> Beaver.Changeset.add_result(i64)
          |> MLIR.Operation.create()

        MLIR.RewriterBase.insert(base, add)
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
