defmodule UpstreamCapabilitiesTest do
  use Beaver.Case, async: true
  alias Beaver.{Composer, Native, Walker}
  alias Beaver.MLIR.{Attribute, CAPI, Context, Operation, Type}
  alias Beaver.MLIR.Dialect.Bufferization

  test "independent context locks allow external workers without a pool" do
    ctx = Context.create(threading: false, thread_safe: true)

    try do
      assert Context.thread_safe?(ctx)
      refute CAPI.beaverContextIsMultithreadingEnabled(ctx) |> Native.to_term()
      assert ctx.thread_pool_owner == nil
      expected = Type.i64(ctx: ctx)

      1..16
      |> Task.async_stream(
        fn _ ->
          for n <- 1..100 do
            assert MLIR.equal?(expected, Type.i64(ctx: ctx))
            assert Attribute.integer(expected, n) |> MLIR.to_string() == "#{n} : i64"
          end
        end,
        max_concurrency: 16
      )
      |> Enum.each(fn {:ok, _} -> :ok end)
    after
      Context.destroy(ctx)
    end

    assert_raise ArgumentError, ~r/requires thread_safe/, fn -> Context.create(thread_safe: false) end
    assert_raise ArgumentError, ~r/thread_pool/, fn -> Context.create(threading: false, thread_pool: :application) end
  end

  for bits <- [32, 64] do
    test "Index, Arith and Func agree on a #{bits}-bit DataLayout", %{ctx: ctx} do
      bits = unquote(bits)

      module =
        ("module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, #{bits}>>} {\n" <>
           "  func.func @sum(%x: index) -> index {\n    %one = arith.constant 1 : index\n    %a = index.add %x, %x\n    %b = arith.addi %a, %one : index\n    return %b : index\n  }\n}\n")
        |> MLIR.Module.create!(ctx: ctx)

      ir =
        module
        |> Composer.append(
          "convert-index-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts"
        )
        |> Composer.run!()
        |> MLIR.to_string()

      assert ir =~ "llvm.func @sum(%arg0: i#{bits}) -> i#{bits}"
      refute ir =~ "unrealized_conversion_cast"
    end
  end

  test "explicit index-bitwidth wins, and no layout defaults to 64", %{ctx: ctx} do
    for {layout, options} <- [
          {"", ""},
          {"attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>}", "{index-bitwidth=64}"}
        ] do
      module =
        ("module #{layout} {\n" <>
           "  func.func @sum(%x: index) -> index {\n    %a = index.add %x, %x\n    return %a : index\n  }\n}\n")
        |> MLIR.Module.create!(ctx: ctx)

      ir = module |> Composer.append("convert-index-to-llvm#{options}") |> Composer.run!() |> MLIR.to_string()
      assert ir =~ "index to i64"
      refute ir =~ "index to i32"
    end
  end

  test "native property metadata includes types and defaults" do
    assert {:ok, load} = MLIR.ODS.Dump.lookup("ptr.load")
    alignment = Enum.find(load["properties"], &(&1["name"] == "alignment"))
    assert alignment["storage_type"] =~ "optional"
    assert alignment["interface_type"] =~ "optional"
    assert is_binary(alignment["constraint"])
    ordering = Enum.find(load["properties"], &(&1["name"] == "ordering"))
    assert ordering["default_value"] =~ "not_atomic"
    refute Enum.any?(load["attributes"] || [], &(&1["name"] == "alignment"))
  end

  test "properties preserve native values and discardable attrs through text and bytecode", %{ctx: ctx} do
    module =
      "module {\n  func.func @load(%p: !ptr.ptr<#ptr.generic_space>) -> i32 {\n    %v = ptr.load %p alignment = 8 {test.note = \"keep\"} : !ptr.ptr<#ptr.generic_space> -> i32\n    return %v : i32\n  }\n}\n"
      |> MLIR.Module.create!(ctx: ctx)

    load = operation_named(module, "ptr.load")
    properties = load |> Operation.properties() |> MLIR.to_string()
    assert properties =~ "alignment = [8]"
    updated = properties |> String.replace("alignment = [8]", "alignment = [16]") |> Attribute.get(ctx: ctx)
    assert {:ok, ^load} = Operation.put_properties(load, updated)
    assert Operation.properties(load) |> MLIR.to_string() =~ "alignment = [16]"
    assert Operation.discardable_attribute(load, "test.note") |> MLIR.to_string() == ~s("keep")
    before = MLIR.to_string(load, generic: true)
    invalid = properties |> String.replace("alignment = [8]", ~s(alignment = "invalid")) |> Attribute.get(ctx: ctx)
    assert {:error, [_ | _]} = Operation.put_properties(load, invalid)
    assert MLIR.to_string(load, generic: true) == before

    for text <- [MLIR.to_string(module), MLIR.to_string(module, generic: true), MLIR.Bytecode.write!(module)] do
      roundtrip = MLIR.Module.create!(text, ctx: ctx) |> MLIR.verify!()
      restored = operation_named(roundtrip, "ptr.load")
      assert MLIR.equal?(Operation.properties(restored), Operation.properties(load))
      assert Operation.discardable_attribute(restored, "test.note") |> MLIR.to_string() == ~s("keep")
    end
  end

  @tag :static_memory_planner
  test "best-fit arena reuses disjoint lifetimes and keeps overlapping buffers separate", %{ctx: ctx} do
    for {overlap, bytes} <- [{false, 16}, {true, 32}] do
      release = "memref.dealloc %a : memref<4xf32>"

      module =
        ("" <>
           "func.func @arena() {\n  %a = memref.alloc() : memref<4xf32>\n  #{if not overlap, do: release}\n  %b = memref.alloc() : memref<4xf32>\n  #{if overlap, do: release}\n  memref.dealloc %b : memref<4xf32>\n  return\n}\n")
        |> MLIR.Module.create!(ctx: ctx)

      ir = module |> Bufferization.static_memory_plan() |> Composer.run!() |> MLIR.to_string()
      assert ir =~ "memref<#{bytes}xi8>"
      assert length(Regex.scan(~r/memref\.view/, ir)) == 2
      assert length(Regex.scan(~r/memref\.alloc\(/, ir)) == 1
      assert length(Regex.scan(~r/memref\.dealloc/, ir)) == 1
    end
  end

  @tag :static_memory_planner
  test "caller-owned arena needs no allocation and invalid arenas report diagnostics", %{ctx: ctx} do
    for {args, succeeds} <- [{"%arena: memref<32xi8>", true}, {"%arena: memref<8xi8>", false}, {"", false}] do
      module =
        ("" <>
           "func.func @arena(#{args}) {\n  %a = memref.alloc() : memref<4xf32>\n  memref.dealloc %a : memref<4xf32>\n  return\n}\n")
        |> MLIR.Module.create!(ctx: ctx)

      result = module |> Bufferization.static_memory_plan(arena_mode: :arg) |> Composer.run()

      if succeeds do
        assert {:ok, planned, _} = result
        ir = MLIR.to_string(planned)
        assert ir =~ "memref.view %arg0"
        refute ir =~ "memref.alloc"
      else
        assert {:error, _} = result
      end
    end
  end

  @tag :static_memory_planner
  test "shared deallocs and unsupported layouts and elements retain their original releases", %{ctx: ctx} do
    module =
      ("" <>
         "func.func @skip(%n: index, %cond: i1) {\n  %a = memref.alloc() : memref<4xf32>\n  %b = memref.alloc(%n) : memref<?xf32>\n  %ac = memref.cast %a : memref<4xf32> to memref<?xf32>\n  %s0 = arith.select %cond, %ac, %b : memref<?xf32>\n  %s1 = arith.select %cond, %b, %ac : memref<?xf32>\n  memref.dealloc %s0 : memref<?xf32>\n  memref.dealloc %s1 : memref<?xf32>\n  %c = memref.alloc() : memref<4xcomplex<f32>>\n  memref.dealloc %c : memref<4xcomplex<f32>>\n  %d = memref.alloc() : memref<4xf32, strided<[2]>>\n  memref.dealloc %d : memref<4xf32, strided<[2]>>\n  return\n}\n")
      |> MLIR.Module.create!(ctx: ctx)

    before = MLIR.to_string(module)
    assert module |> Bufferization.static_memory_plan() |> Composer.run!() |> MLIR.to_string() == before
  end

  @tag :static_memory_planner
  test "arena is released on every return path and unsafe packed element sizes are rejected", %{ctx: ctx} do
    module =
      ("" <>
         "func.func @arena(%cond: i1) {\n  %a = memref.alloc() : memref<4xf32>\n  memref.dealloc %a : memref<4xf32>\n  cf.cond_br %cond, ^left, ^right\n^left:\n  return\n^right:\n  return\n}\n")
      |> MLIR.Module.create!(ctx: ctx)

    ir = module |> Bufferization.static_memory_plan() |> Composer.run!() |> MLIR.to_string()
    assert length(Regex.scan(~r/memref\.alloc\(/, ir)) == 1
    assert length(Regex.scan(~r/memref\.dealloc/, ir)) == 2

    packed =
      ("" <>
         "func.func @packed() {\n  %a = memref.alloc() : memref<8xi1>\n  memref.dealloc %a : memref<8xi1>\n  return\n}\n")
      |> MLIR.Module.create!(ctx: ctx)

    assert {:error, _} = packed |> Bufferization.static_memory_plan() |> Composer.run()
  end

  defp operation_named(module, name) do
    {_, found} =
      Walker.prewalk(module, nil, fn
        %Operation{} = op, acc -> {op, if(Operation.name(op) == name, do: op, else: acc)}
        node, acc -> {node, acc}
      end)

    found
  end
end
