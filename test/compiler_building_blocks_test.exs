defmodule Beaver.CompilerBuildingBlocksTest do
  use Beaver.Case, async: true
  alias Beaver.{Composer, MLIR, Native}
  alias Beaver.MLIR.{Attribute, CABI, Context, ExecutionEngine, IRBatch, Module, Operation, SymbolTable}
  alias Beaver.MLIR.IRBatch.{Builder, Expressions}
  alias Beaver.MLIR.Template.Library
  alias Beaver.Pattern.PDL

  test "immutable CFGs join multiple values and execute every dispatch arm", %{ctx: ctx} do
    initial = Builder.new(data: %{language: :test}) |> Builder.start_block(:entry)
    {zero, b} = Builder.constant(initial, :i64, 0, scope: :entry)
    assert {^zero, ^b} = Builder.constant(b, :i64, 0)
    assert initial.operations == []
    {positive, b} = Builder.compare(b, :sge, :x, zero)

    {[magnitude, extra], b} =
      Builder.guarded(b, positive, [:x, zero], [:i64, :i64], fn acc ->
        {negative, acc} = Builder.binary(acc, "subi", [zero, :x], :i64)
        {[negative, zero], acc}
      end)

    {sum, b} = Builder.binary(b, "addi", [magnitude, extra], :i64)

    b =
      Builder.dispatch(b, :selector, :i32, :default, [{0, :zero, 0}, {1, :one, 1}], fn n, acc ->
        {value, acc} = Builder.constant(acc, :i64, n)
        {result, acc} = Builder.binary(acc, "addi", [sum, value], :i64)
        Builder.end_block(acc, {:return, [result]})
      end)

    b = b |> Builder.start_block(:default) |> Builder.end_block({:return, [zero]})
    assert b.data == %{language: :test}
    spec = run_spec(ctx, b, x: :i64, selector: :i32)

    Native.with_resource(IRBatch.build_module!(ctx, [spec]), &Module.destroy/1, fn module ->
      with_jit(module, fn jit ->
        for {x, selector, expected} <- [{7, 0, 7}, {-7, 1, 8}, {7, 2, 0}] do
          assert invoke(jit, [Native.I64.make(x), Native.I32.make(selector)]) == expected
        end
      end)
    end)
  end

  test "block constants never escape their scope and malformed CFG assembly is rejected" do
    b = Builder.new() |> Builder.start_block(:entry)
    {entry, b} = Builder.constant(b, :i64, 9)
    b = b |> Builder.end_block({:br, :next}) |> Builder.start_block(:next)
    {local, b} = Builder.constant(b, :i64, 9)
    refute entry == local
    assert {^local, ^b} = Builder.constant(b, :i64, 9)
    assert_raise ArgumentError, fn -> Builder.constant(b, :i64, 10, scope: :entry) end
    assert_raise ArgumentError, fn -> Builder.finish(b) end
    assert_raise ArgumentError, fn -> Builder.start_block(b, :other) end
    assert_raise ArgumentError, fn -> b |> Builder.end_block({:return, []}) |> Builder.start_block(:entry) end
    assert_raise ArgumentError, fn -> Builder.guarded(b, :cond, [], [:i64], fn x -> {[], x} end) end

    assert_raise ArgumentError, fn ->
      Builder.dispatch(b, :selector, :i32, :default, [{0, :case, nil}], fn _, x -> x end)
    end
  end

  test "nested expressions evaluate effectful operands once in left-to-right order", %{ctx: ctx} do
    source =
      "module { func.func @next(%p: !llvm.ptr) -> i64 { %v = llvm.load %p : !llvm.ptr -> i64 %one = arith.constant 1 : i64 %n = arith.addi %v, %one : i64 llvm.store %n, %p : i64, !llvm.ptr return %v : i64 } }"

    b = Builder.new() |> Builder.start_block(:entry)
    {one, b} = Builder.constant(b, :i64, 1)
    {ptr, b} = Builder.value(b, "llvm.alloca", [one], :ptr, elem_type: {:type, :i64})
    b = Builder.operation(b, "llvm.store", [one, ptr], [], [], nil)
    next = Builder.expression(:call, ["next", [ptr], [:i64]])
    difference = Builder.expression(:binary, ["subi", [next, next], :i64])
    b = Builder.end_block(b, {:return, [difference]})
    finalize = fn module -> Module.with_module(ctx, source, &Module.import_symbols!(module, &1)) end
    module = IRBatch.build_module!(ctx, [run_spec(ctx, b, [])], finalize: finalize)
    Native.with_resource(module, &Module.destroy/1, &with_jit(&1, fn jit -> assert invoke(jit, []) == -1 end))
  end

  test "expression libraries infer shared declarations across batches and reject inconsistent calls", %{ctx: ctx} do
    alias IRBatch.Expressions, as: E
    specs = for i <- 1..300, do: E.forward("f#{i}", "external", [:i64], :i64)
    specs = specs ++ E.memory_accessors("read", "write", :i64)
    body = E.source(specs)

    Module.with_module(ctx, "module {" <> body <> "}", fn module ->
      signatures = SymbolTable.function_signatures(module)
      assert map_size(signatures) == 303
      assert signatures["external"] == {["i64"], ["i64"]}
      assert signatures["write"] == {["!llvm.ptr", "i64", "i64"], []}
    end)

    bad = E.forward("bad", "external", [:i32], :i64)
    assert_raise ArgumentError, fn -> E.source([hd(specs), bad]) end
  end

  test "wide negative constants remain signed when compared", %{ctx: ctx} do
    negative = Expressions.int(-9_223_372_036_854_775_808, :i128)
    value = Expressions.cast("extui", Expressions.cmp(:slt, negative, Expressions.int(0, :i128)), :i64)
    spec = Expressions.function("run", [], :i64, value) |> Map.put(:attributes, "llvm.emit_c_interface": :unit)
    Module.with_module(IRBatch.build_module!(ctx, [spec]), &with_jit(&1, fn jit -> assert invoke(jit, []) == 1 end))
  end

  test "template revisions cache portable data while each use owns independent IR", %{ctx: ctx} do
    source = "module { func.func @identity(%x: i64) -> i64 { return %x : i64 } }"
    key = {__MODULE__, make_ref()}
    on_exit(fn -> :persistent_term.erase({Library, key}) end)
    library = Library.fetch(key, 1, fn -> source end)
    assert Library.fetch(key, 1, fn -> flunk("unchanged revision rebuilt") end) == library
    assert library.signatures == %{"identity" => {["i64"], ["i64"]}}

    Library.with_module(library, ctx, fn module ->
      Operation.put_attribute(Operation.from_module(module), "test.changed", Attribute.unit(ctx: ctx))
    end)

    Context.with_context(fn other ->
      Library.with_module(library, other, fn module -> refute MLIR.to_string(module) =~ "test.changed" end)

      Module.with_module(other, fn target ->
        Library.declare!(library, target, symbols: ["identity"])
        Library.import!(library, target, symbols: ["identity"], verify_type: true)
        assert MLIR.to_string(MLIR.verify!(target)) =~ "return"
      end)
    end)

    replacement = Library.fetch(key, 2, fn -> String.replace(source, "i64", "i32") end)
    refute replacement.digest == library.digest
    assert replacement.signatures == %{"identity" => {["i32"], ["i32"]}}
  end

  test "template declarations preserve definitions and selection imports transitive dependencies", %{ctx: ctx} do
    source =
      "module { func.func @entry(%x: i64) -> i64 { %r = func.call @helper(%x) : (i64) -> i64 return %r : i64 } func.func private @helper(%x: i64) -> i64 { return %x : i64 } func.func @unused() { return } }"

    library = Library.new(source)
    user = "module { func.func @run(%x: i64) -> i64 { %r = func.call @entry(%x) : (i64) -> i64 return %r : i64 } }"
    declared = Library.with_declarations(library, user)
    assert declared == Library.with_declarations(library, declared)
    assert source == Library.with_declarations(library, source)

    Module.with_module(ctx, declared, fn target ->
      before = MLIR.to_string(target)
      Library.import!(library, target, select: fn _ -> false end)
      assert MLIR.to_string(target) == before
      Library.import!(library, target, select: &(&1 == "entry"), verify_type: true)
      assert Map.keys(SymbolTable.function_signatures(MLIR.verify!(target))) |> Enum.sort() == ~w(entry helper run)
      assert MLIR.to_string(target) =~ "return %arg0"
    end)
  end

  test "function mapping keeps bindings intact and expands each CFG position", %{ctx: ctx} do
    spec = %{
      name: "mapped",
      inputs: [term: :term],
      results: :term,
      blocks: [[name: :entry, body: [:next]], [name: :done, args: [result: :term], body: [:return]]]
    }

    expand = fn
      :next, {0, 0} -> [{:br, :done, [:term]}]
      :return, {1, 0} -> [{:return, [:result]}]
    end

    mapped = IRBatch.map_function(spec, fn :term -> :i64 end, expand)
    assert mapped.inputs == [term: :i64]
    assert mapped.blocks |> List.last() |> Map.fetch!(:args) == [result: :i64]
    Native.with_resource(IRBatch.build_module!(ctx, [mapped]), &Module.destroy/1, &MLIR.verify!/1)
  end

  test "declaration and transitive import conflicts leave the target unchanged", %{ctx: ctx} do
    source =
      "module { func.func @entry(%x: i64) -> i64 { %r = func.call @helper(%x) : (i64) -> i64 return %r : i64 } func.func private @helper(%x: i64) -> i64 { return %x : i64 } }"

    target = "module { func.func private @entry(i64) -> i64 func.func private @helper(i32) -> i32 }"

    Module.with_module(ctx, source, fn source ->
      Module.with_module(ctx, target, fn target ->
        before = MLIR.to_string(target)
        types = [{"new", {["i64"], ["i64"]}}, {"helper", {["i64"], ["i64"]}}]
        assert_raise ArgumentError, fn -> SymbolTable.check_declarations!(target, types) end
        assert_raise ArgumentError, fn -> SymbolTable.ensure_declarations!(target, types) end

        assert_raise ArgumentError, fn ->
          Module.import_symbols!(target, source, symbols: ["entry"], verify_type: true)
        end

        assert MLIR.to_string(target) == before

        assert_raise ArgumentError, fn ->
          SymbolTable.ensure_declarations!(target, [{"new", "() -> i32"}, {"new", "() -> i64"}])
        end

        assert MLIR.to_string(target) == before
        SymbolTable.ensure_declarations!(target, [{"new", "() -> i32"}])
        assert SymbolTable.function_signatures(target)["new"] == {[], ["i32"]}
      end)
    end)
  end

  test "PDL selection includes new matching symbols introduced by another rule", %{ctx: ctx} do
    entries = [rename_rule("start", "middle"), rename_rule("middle", "finish")]
    key = {__MODULE__, make_ref()}
    on_exit(fn -> :persistent_term.erase({PDL, key}) end)
    rules = PDL.fetch(key, 1, fn -> entries end)
    assert rules == PDL.fetch(key, 1, fn -> flunk("unchanged rules rebuilt") end)
    refute PDL.new(Enum.reverse(entries)).digest == rules.digest
    refute PDL.applicable?(rules, ["unrelated"])

    source =
      "module { func.func private @start(i64) -> i64 func.func private @middle(i64) -> i64 func.func private @finish(i64) -> i64 func.func @user(%x: i64) -> i64 { %r = func.call @start(%x) : (i64) -> i64 return %r : i64 } }"

    Module.with_module(ctx, source, fn module ->
      PDL.apply!(rules, module)
      assert SymbolTable.referenced_symbols(MLIR.verify!(module)) == ["finish"]
    end)

    Context.with_context(fn other ->
      Module.with_module(other, source, fn module ->
        assert SymbolTable.referenced_symbols(PDL.apply!(rules, module)) == ["finish"]
      end)
    end)
  end

  test "C ABI wrappers copy aggregates and zero unwritten out slots", %{ctx: ctx} do
    params = [{:aggregate, "i64"}, {:out, "i64"}]
    assert CABI.value_signature(params, "i32") == {["i64"], ["i32", "i64"]}

    raw =
      "module { func.func @raw(%input: !llvm.ptr, %output: !llvm.ptr) -> i32 { %value = llvm.load %input : !llvm.ptr -> i64 llvm.store %value, %output : i64, !llvm.ptr %zero = arith.constant 0 : i32 return %zero : i32 } }"

    failed =
      "module { func.func @raw(%input: !llvm.ptr, %output: !llvm.ptr) -> i32 { %status = arith.constant 2 : i32 return %status : i32 } }"

    entry =
      "func.func @run(%x: i64) -> i64 attributes {llvm.emit_c_interface} { %s, %v = func.call @wrapped(%x) : (i64) -> (i32, i64) %wide = arith.extui %s : i32 to i64 %r = arith.addi %v, %wide : i64 return %r : i64 }"

    library =
      Library.new("module {\n" <> CABI.wrapper_source("wrapped", "raw", params, return_type: "i32") <> entry <> "\n}")

    for {implementation, expected} <- [{raw, 42}, {failed, 2}] do
      Library.with_module(library, ctx, fn module ->
        Module.with_module(ctx, implementation, &Module.import_symbols!(module, &1, verify_type: true))
        with_jit(module, fn jit -> assert invoke(jit, [Native.I64.make(42)]) == expected end)
      end)
    end

    for {parameters, opts} <- [{[], []}, {[{:out_ptr, "i64", zero_init: false}], [stack_save: false]}] do
      source = CABI.wrapper_source("void_wrapper", "void_raw", parameters, opts)
      assert %Library{} = Library.new("module {\n" <> source <> "\n}")
      refute source =~ "stacksave"
      refute source =~ "mlir.zero"
    end
  end

  defp rename_rule(from, to), do: PDL.Source.retarget("rename_#{from}", from, to, ["i64"])

  test "structured PDL retains result order and rejects mismatched result types", %{ctx: ctx} do
    match = [{[:a, :b], "func.call", [:x, :y], ["i64", "i64"], [callee: "@pair"]}]
    rule = PDL.Source.pattern("swap", "pair", 1, match, [], [:y, :x])

    source =
      "func.func private @pair(i64, i64) -> (i64, i64)\nfunc.func @run(%x: i64, %y: i64) -> i64 attributes {llvm.emit_c_interface} { %a, %b = func.call @pair(%x, %y) : (i64, i64) -> (i64, i64) %r = arith.subi %a, %b : i64 return %r : i64 }"

    Module.with_module(ctx, source, fn module ->
      PDL.new([rule]) |> PDL.apply!(module)
      assert SymbolTable.referenced_symbols(MLIR.verify!(module)) == []
      with_jit(module, fn jit -> assert invoke(jit, [Native.I64.make(17), Native.I64.make(3)]) == -14 end)
    end)

    bad = [{[:a, :b], "func.call", [:x], ["i64"], [callee: "@pair"]}]

    assert_raise ArgumentError, "PDL result/type count mismatch", fn ->
      PDL.Source.pattern("bad", "pair", 1, bad, [], [:a])
    end
  end

  @tag :tmp_dir
  test "header bindings retain aggregate pointers, output slots and namespace validation", %{tmp_dir: dir} do
    alias CABI.Provider
    header = Path.join(dir, "api.h")
    File.write!(header, "int demo_echo(uint64_t n, Payload value, uint64_t *out);\n")

    spec =
      Provider.read!(
        header: header,
        c_return: "int",
        c_prefix: "demo_",
        return_type: "i32",
        types: %{"uint64_t" => "i64"},
        parameters: %{"Payload" => {:aggregate, "i64"}, "uint64_t *" => {:out, "i64"}},
        aggregates: ["Payload"],
        raw_prefix: "raw.",
        value_prefix: "value.",
        namespace_prefix: "host_",
        prelude: "",
        export: "EXPORT",
        invalid: "INVALID"
      )

    assert Provider.functions(spec) == ["echo"]
    assert Provider.argument_types(spec, "echo") == ["i64", "!llvm.ptr", "!llvm.ptr"]
    assert Provider.value_signature(spec, "echo") == {["i64", "i64"], ["i32", "i64"]}
    assert Provider.bindings(spec, "one", ["echo"]) == %{"host_one_echo" => ["host_one_echo", "demo_echo"]}
    adapter = Provider.adapter_source(spec, "one")
    assert adapter =~ "const Payload * a1" and adapter =~ "if (!a1) return INVALID;"
    assert adapter =~ "return demo_echo(a0, *a1, a2)"
    assert_raise ArgumentError, fn -> Provider.adapter_source(spec, "bad-namespace") end
    assert_raise FunctionClauseError, fn -> Provider.raw_symbol(spec, "missing") end
  end

  defp run_spec(ctx, b, inputs) do
    %{name: "run", inputs: inputs, results: [:i64], blocks: Builder.finish(b)}
    |> Map.put(:attributes, "llvm.emit_c_interface": Attribute.unit(ctx: ctx))
  end

  defp with_jit(module, fun) do
    passes = ~w(convert-arith-to-llvm convert-cf-to-llvm convert-func-to-llvm reconcile-unrealized-casts)
    Enum.reduce(passes, module, &Composer.append(&2, &1)) |> Composer.run!()
    Native.with_resource(ExecutionEngine.create!(module), &ExecutionEngine.destroy/1, fun)
  end

  defp invoke(jit, args), do: Native.to_term(tap(Native.I64.make(0), &ExecutionEngine.invoke!(jit, "run", args, &1)))
end
