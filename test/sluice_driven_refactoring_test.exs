defmodule Beaver.SluiceDrivenRefactoringTest.HygienicMacroHelper do
  @moduledoc false
  defmacro def_pinned_unary_op(op_name) do
    quote do
      Beaver.Slang.defop(unquote(op_name)(input = ^i64_or_i32),
        results: [output: ^i64_or_i32],
        layer: :frontend,
        metadata: %{wrapped: true}
      )
    end
  end
end

defmodule Beaver.SluiceDrivenRefactoringTest.LayeredSlangDialect do
  @moduledoc false
  use Beaver.Slang, name: "sluice_slang"
  import Beaver.SluiceDrivenRefactoringTest.HygienicMacroHelper

  defconstraint i64_or_i32 do
    any_of([Beaver.MLIR.Type.i64(), Beaver.MLIR.Type.i32()])
  end

  def_pinned_unary_op(:box_int)

  Beaver.Slang.defops(
    [
      {:add, [lhs: :i64, rhs: :i64], [sum: :i64], [], [layer: :core, metadata: %{opcode: 1}]},
      {:widen, [val: :i64_or_i32], [out: :i64], [], [layer: :core, metadata: %{opcode: 3}]},
      {:const_int, [], [out: :i64], [value: "#builtin.integer"], layer: :core, metadata: %{opcode: 2}},
      %{
        name: :halt,
        operands: [code: :i32],
        results: [],
        traits: [:terminator],
        layer: :exit,
        metadata: %{exit_kind: :halt}
      }
    ],
    layer: :default_layer,
    metadata: %{suite: :sluice}
  )
end

defmodule Beaver.SluiceDrivenRefactoringTest do
  use Beaver.Case, async: true
  alias Beaver.{Changeset, Composer, MLIR, Native}

  alias Beaver.MLIR.{Block, Context, ExecutionEngine, Identifier, IRBatch}
  alias Beaver.MLIR.{Location, Module, Operation, Region, RewriterBase}
  alias Beaver.MLIR.{Safepoint, SymbolTable, Template, Type, TypeConverter}

  alias Beaver.MLIR.ConversionPatternRewriter, as: CPR
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Target.LLVMIR
  alias Beaver.Pattern.Native.Program
  alias Beaver.Slang
  alias Beaver.SluiceDrivenRefactoringTest.LayeredSlangDialect, as: Dialect
  alias Beaver.Walker

  describe "1. CAPI wrappers and with_* resource helpers" do
    test "Context.with_context and Module.with_module manage lifecycle and options" do
      rendered =
        Context.with_context([allow_unregistered: true], fn ctx ->
          assert Context.allow_unregistered_dialects?(ctx)

          Module.with_module(ctx, fn empty_mod ->
            assert MLIR.verify!(empty_mod)
            body = Module.body(empty_mod)
            assert Enum.to_list(Block.operations(body)) == []
          end)

          ir = "module {\n  \"unreg.marker\"() {tag = \"ok\"} : () -> ()\n}\n"

          Module.with_module(ctx, ir, fn mod ->
            assert MLIR.verify!(mod)
            MLIR.to_string(mod)
          end)
        end)

      assert rendered =~ "unreg.marker"
    end

    test "Region, Block, Identifier, and Operation navigation/attribute wrappers", %{ctx: ctx} do
      loc = Location.unknown(ctx: ctx)
      region = Region.create()

      try do
        assert Region.first_block(region) == nil
        assert Enum.to_list(Region.blocks(region)) == []
        b0 = Region.append_block(region, [{Type.i32(ctx: ctx), loc}])
        b1 = Region.append_block(region, [Type.i64(ctx: ctx)], location: loc)
        assert Region.blocks(region) |> Enum.count() == 2
        assert Block.equal?(Region.first_block(region), b0)
        refute Block.equal?(b0, b1)
        assert Block.parent_region(b0) != nil
        assert Block.parent_operation(b0) == nil
        extra_arg = Block.add_arg!(b0, Type.i1(ctx: ctx), location: loc)
        assert MLIR.to_string(MLIR.Value.type(extra_arg)) == "i1"
        named = Block.add_named_args!(b1, [lhs: Type.i32(ctx: ctx), rhs: Type.i64(ctx: ctx)], location: loc)
        assert Map.keys(named) |> Enum.sort() == [:lhs, :rhs]
        assert length(Block.arguments(b1)) == 3
      after
        Region.destroy(region)
      end

      [id_a, id_b, id_c] = Enum.map(~w(sym_name sym_name other_attr), &Identifier.get(&1, ctx: ctx))
      assert Identifier.to_string(id_a) == "sym_name"
      assert Identifier.equal?(id_a, id_b)
      refute Identifier.equal?(id_a, id_c)
      Context.allow_unregistered_dialects(ctx)

      ir =
        "module {\n  func.func @demo(%x: i64) -> i64 {\n    %0 = \"demo.first\"(%x) {a = 1 : i64, b = \"two\", c = true} : (i64) -> i64\n    %1 = \"demo.second\"(%0) : (i64) -> i64\n    return %1 : i64\n  }\n}\n"

      Module.with_module(ctx, ir, fn mod ->
        mod_op = Operation.from_module(mod)
        mod_body = Module.body(mod_op)
        [func_op] = Enum.to_list(Block.operations(mod_body))
        [entry] = func_op |> Walker.regions() |> Enum.at(0) |> Region.blocks() |> Enum.to_list()
        [op0, op1, term] = Enum.to_list(Block.operations(entry))
        assert Operation.name(Block.first_operation(entry)) == "demo.first"
        assert Operation.name(Block.terminator(entry)) == "func.return"
        assert Operation.next_in_block(term) == nil
        assert Operation.name(Operation.next_in_block(op0)) == "demo.second"
        assert Operation.block(op0) != nil
        assert Operation.name(Operation.parent_op(op0)) == "func.func"
        assert Operation.name(Operation.root(op0)) == "builtin.module"
        assert Operation.name(Operation.enclosing_symbol_table(op0)) == "builtin.module"
        Operation.copy_attributes(op0, op1, only: [:a, "b"])
        rendered1 = MLIR.to_string(op1)
        assert_text(rendered1, ["a = 1 : i64", ~s(b = "two")], ["c = true"])
        Operation.copy_attributes(op0, term, except: [:a, :b])
        assert MLIR.to_string(term) =~ "c = true"
      end)
    end
  end

  describe "2. Changeset, RewriterBase structured helpers, and SymbolTable" do
    test "Changeset accepts shorthand result types and attribute literals", %{ctx: ctx} do
      loc = Location.unknown(ctx: ctx)

      op =
        %Changeset{name: "arith.constant", location: loc}
        |> Changeset.add_argument(value: {:i64, 99})
        |> Changeset.add_argument(tag: {:string, "lit"}, marker: :unit, enabled: true)
        |> Changeset.add_argument(callee: {:symbol, "target_fn"}, segs: {:dense_i64, [2, 4]})
        |> Changeset.add_result(:i64)
        |> Operation.create()

      try do
        text = MLIR.to_string(op, generic: true)

        assert_text(
          text,
          ["99 : i64", ~s(tag = "lit"), "marker", "enabled = true", "callee = @target_fn", "array<i64: 2, 4>"],
          []
        )
      after
        Operation.destroy(op)
      end
    end

    test "RewriterBase emit, scf_if, pack_struct, unpack_struct, and SymbolTable.ensure_declaration!",
         %{ctx: ctx} do
      Context.allow_unregistered_dialects(ctx)

      plan =
        Plan.new(mode: :partial)
        |> Plan.add_legal_dialects(["builtin", "func", "arith", "scf", "llvm"])
        |> Plan.add_illegal_op("demo.select_and_roundtrip")
        |> Plan.add_conversion_pattern("demo.select_and_roundtrip", fn op, [cond_v, a_v, b_v], rw ->
          helper_ref =
            SymbolTable.ensure_declaration!(op, "ext_helper", "(i64) -> i64", op: "func.func", visibility: "private")

          assert MLIR.to_string(helper_ref) =~ "func.func private @ext_helper(i64) -> i64"

          assert_raise ArgumentError, ~r/already declared with incompatible function_type/, fn ->
            SymbolTable.ensure_declaration!(op, "ext_helper", "(i32) -> i32", op: "func.func", verify_type: true)
          end

          then_branch = fn rw, loc -> [CPR.emit(rw, loc, "arith.addi", [a_v, a_v], [:i64])] end

          else_branch = fn rw, loc ->
            c10 = CPR.emit(rw, loc, "arith.constant", [value: {:i64, 10}], [:i64])
            [CPR.emit(rw, loc, "arith.addi", [b_v, c10], [:i64])]
          end

          [selected] = CPR.scf_if(rw, op, cond_v, [:i64], then_branch, else_branch)
          tag = RewriterBase.emit(rw, op, "arith.constant", [value: {:i32, 7}], [:i32])
          struct_ty = "!llvm.struct<(i64, i32)>"
          packed = CPR.pack_struct(rw, op, struct_ty, [selected, tag])
          [unpacked_val, _unpacked_tag] = CPR.unpack_struct(rw, op, packed, 2)
          CPR.replace_op(rw, op, [unpacked_val])
          :ok
        end)

      ir =
        "module {\n  func.func @run(%flag: i1, %a: i64, %b: i64) -> i64 {\n    %r = \"demo.select_and_roundtrip\"(%flag, %a, %b) : (i1, i64, i64) -> i64\n    return %r : i64\n  }\n}\n"

      Module.with_module(ctx, ir, fn mod ->
        assert {:ok, ^mod, []} = Plan.run(plan, mod)
        assert MLIR.verify!(mod)
        rendered = MLIR.to_string(mod)

        assert_text(
          rendered,
          ["func.func private @ext_helper(i64) -> i64", "scf.if", "llvm.insertvalue", "llvm.extractvalue"],
          ["demo.select_and_roundtrip"]
        )
      end)
    end
  end

  describe "3. Multi-block IRBatch, inline attributes, and MLIR.Template" do
    test "IRBatch.build_module! builds multi-block loop and switch functions that JIT and execute",
         %{ctx: ctx} do
      Context.register_translations(ctx)

      loop_blocks = [
        entry: {[], [constant(:zero, 0), constant(:one, 1), {:br, {:loop, [:zero, :zero]}}]},
        loop:
          {[i: :i64, acc: :i64],
           [
             operation("arith.cmpi", :keep_going, [:i, :n], [:i1], predicate: {:i64, 2}),
             {:cond_br, :keep_going, {:body, []}, {:done, [:acc]}}
           ]},
        body:
          {[],
           [
             operation("arith.addi", :next_i, [:i, :one]),
             operation("arith.addi", :next_acc, [:acc, :next_i]),
             {:br, {:loop, [:next_i, :next_acc]}}
           ]},
        done: {[result: :i64], [{:return, [:result]}]}
      ]

      switch_blocks = [
        entry:
          {[], [{:switch, {:i32, :tag}, {:fallback, [:x]}, [{0, {:add_case, [:x, :y]}}, {1, {:mul_case, [:x, :y]}}]}]},
        add_case: {[a: :i64, b: :i64], [operation("arith.addi", :sum, [:a, :b]), {:return, [:sum]}]},
        mul_case: {[a: :i64, b: :i64], [operation("arith.muli", :prod, [:a, :b]), {:return, [:prod]}]},
        fallback: {[v: :i64], [{:return, [:v]}]}
      ]

      specs =
        for {name, args, blocks} <- [
              {"sum_up_to", [n: :i64], loop_blocks},
              {"dispatch_op", [tag: :i32, x: :i64, y: :i64], switch_blocks}
            ],
            do: {:func, name, args, [:i64], attributes: [{"llvm.emit_c_interface", :unit}], blocks: blocks}

      Module.with_module(IRBatch.build_module!(ctx, specs), fn mod ->
        assert MLIR.verify!(mod)

        ExecutionEngine.with_jit(lower_llvm(mod), fn jit ->
          assert invoke_i64(jit, "sum_up_to", [Native.I64.make(10)]) == 55

          for {tag, expected} <- [{0, 13}, {1, 42}, {9, 6}] do
            args = [Native.I32.make(tag), Native.I64.make(6), Native.I64.make(7)]
            assert invoke_i64(jit, "dispatch_op", args) == expected
          end
        end)
      end)
    end

    test "Template.render! and Template.module! validate tokens and escape strings", %{ctx: ctx} do
      assert Template.check_ident!("valid_name") == "valid_name"
      assert_raise ArgumentError, fn -> Template.check_ident!("invalid name") end

      rendered =
        Template.render!(
          "module {\n  func.func {{fn_sym}}({{arg_ssa}}: {{ty}}) -> {{ty}} attributes {label = {{note}}} {\n  {{entry_blk}}:\n    {{c_ssa}} = arith.constant {{bias}} : {{ty}}\n    {{r_ssa}} = arith.addi {{arg_ssa}}, {{c_ssa}} : {{ty}}\n    return {{r_ssa}} : {{ty}}\n  }\n}\n",
          fn_sym: {:sym, "add_bias"},
          arg_ssa: {:ssa, "x"},
          c_ssa: {:ssa, "bias_v"},
          r_ssa: {:ssa, "res"},
          entry_blk: {:blk, "bb0"},
          ty: {:type, :i64},
          bias: {:int, 17},
          note: {:str, "line1\n\"quoted\""}
        )

      assert_text(rendered, ["func.func @add_bias(%x: i64)", ~s(label = "line1\\n\\"quoted\\"")], [])

      mod =
        Template.module!(
          ctx,
          "module {\n  func.func {{sym}}(%x: i64) -> i64 {\n    return %x : i64\n  }\n}\n",
          sym: {:sym, "id_fn"}
        )

      Module.with_module(mod, fn _ -> assert MLIR.verify!(mod) end)

      assert_raise ArgumentError, ~r/invalid control character in MLIR symbol/, fn ->
        Template.check_symbol!("bad\nsymbol")
      end

      assert_raise ArgumentError, ~r/invalid MLIR SSA/, fn -> Template.check_ssa!("x) -> () {") end
      assert_raise ArgumentError, ~r/invalid MLIR block/, fn -> Template.check_block!("bb0: return") end
    end
  end

  describe "4. Walker.find_ops and Safepoint frame materialization + leaf fast-path" do
    test "Walker.find_ops filters by name, set, and predicate with exclude_root", %{ctx: ctx} do
      ir =
        "module {\n  func.func @f(%a: i64, %b: i64) -> i64 {\n    %c = arith.constant 5 : i64\n    %s = arith.addi %a, %b : i64\n    %m = arith.muli %s, %c : i64\n    return %m : i64\n  }\n}\n"

      Module.with_module(ctx, ir, fn mod ->
        assert length(Walker.find_ops(mod, "arith.addi")) == 1
        assert length(Walker.find_ops(mod, ["arith.addi", "arith.muli"])) == 2
        [func_op] = Walker.find_ops(mod, "func.func")
        refute Enum.any?(Walker.find_ops(func_op, :all, exclude_root: true), &(Operation.name(&1) == "func.func"))
        assert Enum.any?(Walker.find_ops(func_op, :all, exclude_root: false), &(Operation.name(&1) == "func.func"))
        arith_ops = Walker.find_ops(func_op, fn op -> String.starts_with?(Operation.name(op), "arith.") end)
        assert length(arith_ops) == 3
      end)
    end

    test "Safepoint.lower_resumable! materializes native LLVM GEP/store/load with allow_unregistered: false and executes across resumes" do
      Context.with_context([allow_unregistered: false], fn ctx ->
        Context.register_translations(ctx)

        ir =
          "module {\n  func.func private @coro_suspend_marker() -> ()\n  func.func @step_machine(%state: i32, %frame: !llvm.ptr, %input: i64) -> i64 attributes {llvm.emit_c_interface} {\n    %c10 = arith.constant 10 : i64\n    %v1 = arith.addi %input, %c10 : i64\n    %extra = arith.muli %input, %c10 : i64\n    func.call @coro_suspend_marker() : () -> ()\n    %c3 = arith.constant 3 : i64\n    %v2 = arith.muli %v1, %c3 : i64\n    %v2_extra = arith.addi %v2, %extra : i64\n    func.call @coro_suspend_marker() : () -> ()\n    %c5 = arith.constant 5 : i64\n    %v3 = arith.subi %v2_extra, %c5 : i64\n    return %v3 : i64\n  }\n}\n"

        Module.with_module(ctx, ir, fn mod ->
          [step_fn] =
            Walker.find_ops(mod, fn op ->
              Operation.name(op) == "func.func" and MLIR.to_string(op) =~ "@step_machine"
            end)

          assert {:ok, ^step_fn, frame_bytes} =
                   Safepoint.lower_resumable(step_fn,
                     safepoints: ["func.call"],
                     state_arg: 0,
                     frame_arg: 1,
                     slot_stride: :auto,
                     return_frame_bytes: true
                   )

          assert frame_bytes >= 8
          assert MLIR.verify!(mod)
          lowered_text = MLIR.to_string(mod)
          assert_text(lowered_text, ["llvm.getelementptr", "llvm.store", "llvm.load"], ["coro.save", "coro.load"])
          refute MLIR.to_string(step_fn) =~ "@coro_suspend_marker"
        end)

        # Also verify i32 status protocol end-to-end under JIT execution across state 0 -> 1 -> 2 -> 0!
        jit_ir =
          "module {\n  func.func private @yield_point() -> ()\n  func.func @coro_exec(%state: i32, %frame: !llvm.ptr, %out: !llvm.ptr, %input: i64) -> i32 attributes {llvm.emit_c_interface} {\n    %c10 = arith.constant 10 : i64\n    %v1 = arith.addi %input, %c10 : i64\n    func.call @yield_point() : () -> ()\n    %c3 = arith.constant 3 : i64\n    %v2 = arith.muli %v1, %c3 : i64\n    func.call @yield_point() : () -> ()\n    llvm.store %v2, %out : i64, !llvm.ptr\n    %done = arith.constant 0 : i32\n    return %done : i32\n  }\n}\n"

        Module.with_module(ctx, jit_ir, fn exec_mod ->
          [coro_fn] =
            Walker.find_ops(exec_mod, fn op ->
              Operation.name(op) == "func.func" and MLIR.to_string(op) =~ "@coro_exec"
            end)

          assert {^coro_fn, frame_bytes} =
                   Safepoint.lower_resumable!(coro_fn,
                     safepoints: ["func.call"],
                     state_arg: 0,
                     frame_arg: 1,
                     slot_stride: 8,
                     return_frame_bytes: true
                   )

          assert frame_bytes == 8

          ExecutionEngine.with_jit(lower_llvm(exec_mod), fn jit ->
            frame = Native.I64.make(0)
            out = Native.I64.make(0)
            status = Native.I32.make(-1)
            # Resume with unrelated input: each step must reload the previous spill.
            for {state, input, expected_status, slot, expected_value} <- [
                  {0, 4, 1, frame, 14},
                  {1, 999, 2, frame, 42},
                  {2, 999, 0, out, 42}
                ] do
              args = [Native.I32.make(state), pointer(frame), pointer(out), Native.I64.make(input)]
              ExecutionEngine.invoke!(jit, "coro_exec", args, status)
              assert Native.to_term(status) == expected_status
              assert Native.to_term(slot) == expected_value
            end
          end)
        end)
      end)
    end

    test "Safepoint.bounded_leaf? and Safepoint.bundle_leaf! clone acyclic leaf functions and fold poll",
         %{ctx: ctx} do
      Context.allow_unregistered_dialects(ctx)

      ir =
        "module {\n  func.func @fast_add(%a: i64, %b: i64) -> i64 {\n    %poll = \"guest.poll_budget\"() : () -> i1\n    %res = scf.if %poll -> (i64) {\n      %zero = arith.constant 0 : i64\n      scf.yield %zero : i64\n    } else {\n      %sum = arith.addi %a, %b : i64\n      scf.yield %sum : i64\n    }\n    return %res : i64\n  }\n\n  func.func @looping(%n: i64) -> i64 {\n    cf.br ^hdr(%n : i64)\n  ^hdr(%x: i64):\n    %zero = arith.constant 0 : i64\n    %cond = arith.cmpi sgt, %x, %zero : i64\n    cf.cond_br %cond, ^hdr(%zero : i64), ^exit(%x : i64)\n  ^exit(%out: i64):\n    return %out : i64\n  }\n}\n"

      Module.with_module(ctx, ir, fn mod ->
        [fast_fn, loop_fn] = Walker.find_ops(mod, "func.func")
        assert Safepoint.bounded_leaf?(fast_fn, safepoints: ["guest.suspend"])
        refute Safepoint.bounded_leaf?(fast_fn, safepoints: ["guest.poll_budget"])
        refute Safepoint.bounded_leaf?(loop_fn, safepoints: ["guest.suspend"])

        bundle =
          Safepoint.bundle_leaf!(mod, fast_fn,
            safepoints: ["guest.suspend"],
            poll: "guest.poll_budget",
            lower_resumable: false,
            require_leaf: true
          )

        assert bundle.bounded_leaf?
        assert bundle.leaf_symbol == "fast_add_leaf"
        assert MLIR.verify!(mod)
        leaf_text = MLIR.to_string(bundle.leaf_function)
        assert_text(leaf_text, ["@fast_add_leaf"], ["guest.poll_budget"])
        assert MLIR.to_string(fast_fn) =~ "guest.poll_budget"
        # Idempotent when called again
        same_bundle =
          Safepoint.bundle_leaf!(mod, fast_fn,
            safepoints: ["guest.suspend"],
            poll: "guest.poll_budget",
            lower_resumable: false
          )

        assert MLIR.to_string(same_bundle.leaf_function) =~ "@fast_add_leaf"

        assert_raise ArgumentError, ~r/not a bounded leaf/, fn ->
          Safepoint.bundle_leaf!(mod, loop_fn,
            safepoints: ["guest.suspend"],
            lower_resumable: false,
            require_leaf: true
          )
        end
      end)
    end
  end

  describe "5. Conversion.Plan and Pattern.Native.Program zero-callback enhancements" do
    test "Program copy_attributes {:except, _}, {:symbol_ref, _}, and convert_function_type: :infer run with 0 BEAM callbacks",
         %{ctx: ctx} do
      Context.allow_unregistered_dialects(ctx)

      fn_prog =
        Program.new("guest.func",
          rewrite:
            {:replace_op, "func.func",
             operands: [],
             result_types: [],
             copy_attributes: {:except, [:guest_meta]},
             regions: :move,
             convert_function_type: :infer}
        )

      call_prog =
        Program.new("guest.call",
          rewrite:
            {:replace_op, "func.call",
             copy_attributes: {:except, [:target_sym, :drop_tag]}, attributes: [callee: {:symbol_ref, :target_sym}]}
        )

      ret_prog = Program.new("guest.return", rewrite: {:replace_op, "func.return", result_types: []})

      plan =
        Plan.new(mode: :full)
        |> Plan.add_legal_dialects(["builtin", "func", "arith"])
        |> Plan.add_illegal_dialects(["guest"])
        |> Plan.add_conversion_map(["i32"], "i64", recursive: true)
        |> Plan.add_structural_type_conversions([:func])
        |> Plan.add_pattern(fn_prog, version: "1.0")
        |> Plan.add_pattern(call_prog, version: "1.0")
        |> Plan.add_pattern(ret_prog, version: "1.0")

      # Notice: "guest.func" intentionally has NO function_type attribute!
      ir =
        "module {\n  func.func private @helper(i64) -> i64\n  \"guest.func\"() ({\n  ^bb0(%x: i32):\n    %y = \"guest.call\"(%x) {target_sym = \"helper\", drop_tag = 99 : i64, keep_note = \"preserved\"} : (i32) -> i32\n    \"guest.return\"(%y) : (i32) -> ()\n  }) {sym_name = \"entry\", guest_meta = \"stripped\"} : () -> ()\n}\n"

      Module.with_module(ctx, ir, fn mod ->
        assert {{:ok, ^mod, []}, receipt} = Plan.profile(plan, mod)
        assert receipt["beam"]["callback_count"] == 0
        assert {:ok, ^mod, []} = Plan.finish(plan, mod)
        assert MLIR.verify!(mod)
        rendered = MLIR.to_string(mod)

        assert_text(
          rendered,
          ["func.func @entry(%arg0: i64) -> i64", "call @helper(%arg0)", ~s(keep_note = "preserved")],
          ["guest_meta", "drop_tag", "target_sym"]
        )
      end)
    end

    test "multi-root 4-arity conversion callback, Plan.finish residual cast detection, and c_abi_call verify_declaration",
         %{ctx: ctx} do
      Context.allow_unregistered_dialects(ctx)

      plan =
        Plan.new(mode: :partial)
        |> Plan.add_legal_dialects(["builtin", "func", "arith"])
        |> Plan.add_illegal_ops(["my.double_a", "my.double_b"])
        |> Plan.add_conversion_map(["i32"], "i64")
        |> Plan.add_structural_type_conversions([:func])
        |> Plan.add_conversion_pattern(["my.double_a", "my.double_b"], fn op, [v], rw, converter ->
          [orig_res_ty] = Operation.result_types(op)
          target_ty = TypeConverter.convert!(converter, orig_res_ty)
          sum = CPR.emit(rw, op, "arith.addi", [v, v], [target_ty])
          CPR.replace_op(rw, op, [sum])
          :ok
        end)

      ir =
        "module {\n  func.func @calc(%x: i32) -> i32 {\n    %0 = \"my.double_a\"(%x) : (i32) -> i32\n    %1 = \"my.double_b\"(%0) : (i32) -> i32\n    return %1 : i32\n  }\n}\n"

      Module.with_module(ctx, ir, fn mod ->
        assert ^mod = Plan.finish!(plan, mod)
        assert MLIR.to_string(mod) =~ "func.func @calc(%arg0: i64) -> i64"
      end)

      residual_ir =
        "module {\n  func.func @leftover(%x: i32) -> i64 {\n    %c = builtin.unrealized_conversion_cast %x : i32 to i64\n    return %c : i64\n  }\n}\n"

      Module.with_module(ctx, residual_ir, fn mod ->
        assert {:error, msg} = Plan.finish(Plan.new(mode: :partial), mod)
        assert msg =~ "residual builtin.unrealized_conversion_cast"
      end)

      abi_plan =
        Plan.new(mode: :full)
        |> Plan.add_legal_dialects(["builtin", "llvm"])
        |> Plan.add_illegal_dialect("ext")
        |> Plan.add_c_abi_call_pattern("ext.op", "llvm.call",
          callee: "@runtime_fn",
          return_type: "i32",
          params: [{:in, "i64"}],
          verify_declaration: true
        )

      mismatch_ir =
        "module {\n  llvm.func @runtime_fn(!llvm.ptr) -> i64\n  llvm.func @entry(%x: i64) -> i32 {\n    %s = \"ext.op\"(%x) : (i64) -> i32\n    llvm.return %s : i32\n  }\n}\n"

      Module.with_module(ctx, mismatch_ir, fn mod -> assert {:error, _diagnostics} = Plan.run(abi_plan, mod) end)
    end
  end

  describe "6. Target.LLVMIR lazy multi-unit cache, shared linking, and ExecutionEngine symbol fallbacks" do
    @tag :tmp_dir
    test "compile_units! skips source callbacks on warm hit, recovers missing .o, links shared library, and binds fallback symbols",
         %{ctx: ctx, tmp_dir: tmp_dir} do
      platform = LLVMIR.host_platform()
      assert platform =~ "-"
      shared_ext = LLVMIR.shared_library_extension()
      assert shared_ext in [".dylib", ".so", ".dll"]
      desc1 = LLVMIR.target_descriptor(opt_level: 2)
      desc2 = LLVMIR.target_descriptor(opt_level: 2)
      desc3 = LLVMIR.target_descriptor(opt_level: 0)
      assert desc1 == desc2 and desc1 != desc3
      counter = :atomics.new(2, signed: false)
      cache_dir = Path.join(tmp_dir, "unit_cache")

      units =
        for {id, name, op, constant, index} <- [
              {"math_add", "unit_add_ten", "add", 10, 1},
              {"math_mul", "unit_mul_three", "mul", 3, 2}
            ] do
          %{
            id: id,
            key: "v1_#{id}",
            source: fn ->
              :atomics.add(counter, index, 1)

              "module {\n  llvm.func @#{name}(%x: i32) -> i32 {\n    %c = llvm.mlir.constant(#{constant} : i32) : i32\n    %r = llvm.#{op} %x, %c : i32\n    llvm.return %r : i32\n  }\n}\n"
            end
          }
        end

      cold = LLVMIR.compile_units!(units, cache_dir: cache_dir, opt_level: 2)
      assert for(i <- 1..2, do: :atomics.get(counter, i)) == [1, 1]
      assert Enum.map(cold, & &1.cached?) == [false, false]
      assert Enum.all?(cold, &(File.regular?(&1.path) and byte_size(&1.object) > 64))
      assert File.regular?(Path.join([cache_dir, "objects", "v1_math_add", "manifest.json"]))
      # Warm hit: zero source callback invocations, zero MLIR context needed!
      warm = LLVMIR.compile_units!(units, cache_dir: cache_dir, opt_level: 2)
      assert for(i <- 1..2, do: :atomics.get(counter, i)) == [1, 1]
      assert Enum.map(warm, & &1.cached?) == [true, true]
      assert Enum.map(warm, & &1.sha256) == Enum.map(cold, & &1.sha256)
      # Corrupt cache recovery: delete only the second unit's .o file
      File.rm!(Enum.at(warm, 1).path)
      recovered = LLVMIR.compile_units!(units, cache_dir: cache_dir, opt_level: 2)
      assert for(i <- 1..2, do: :atomics.get(counter, i)) == [1, 2]
      assert Enum.map(recovered, & &1.cached?) == [true, false]
      # Link both unit objects into a shared library
      shared_path = Path.join(tmp_dir, "libmath_units" <> shared_ext)
      assert ^shared_path = LLVMIR.link_shared!(recovered, shared_path)
      assert File.regular?(shared_path)
      # Invoke via ExecutionEngine.with_jit using fallback symbol list
      caller_ir =
        "module {\n  func.func private @ext_add(i32) -> i32\n  func.func private @ext_mul(i32) -> i32\n  func.func @pipeline(%x: i32) -> i32 attributes {llvm.emit_c_interface} {\n    %a = func.call @ext_add(%x) : (i32) -> i32\n    %b = func.call @ext_mul(%a) : (i32) -> i32\n    return %b : i32\n  }\n}\n"

      Module.with_module(ctx, caller_ir, fn mod ->
        mod = Composer.append(mod, "convert-func-to-llvm") |> Composer.run!()

        symbols = %{
          "ext_add" => {shared_path, ["nonexistent_add_sym", "unit_add_ten"]},
          "ext_mul" => {shared_path, ["missing_a", "missing_b", "unit_mul_three"]}
        }

        ExecutionEngine.with_jit(mod, [symbols: symbols], fn jit ->
          ret = Native.I32.make(0)
          ExecutionEngine.invoke!(jit, "pipeline", [Native.I32.make(4)], ret)
          assert Native.to_term(ret) == 42
        end)
      end)
    end
  end

  describe "7. Beaver.Slang hygienic pins, table-driven defops, layers, metadata, and schema digest" do
    test "loads hygienic macro-pinned and table-driven Slang ops, exposes schema digest, and verifies layers",
         %{ctx: ctx} do
      digest = Dialect.__slang_schema_digest__()
      assert String.starts_with?(digest, "sha256:") and byte_size(digest) == 71
      assert Slang.schema_digest(Dialect) == digest

      qualified = &Enum.map(&1, fn name -> "sluice_slang." <> name end)
      assert Dialect.__slang_operations__() == qualified.(~w(box_int add widen const_int halt))
      assert Dialect.__slang_operations__(:core) == qualified.(~w(add widen const_int))

      assert Dialect.__slang_operations__(:frontend) == ["sluice_slang.box_int"]
      assert Dialect.__slang_operations__(:exit) == ["sluice_slang.halt"]
      assert Dialect.__slang_layer__("box_int") == :frontend
      assert Dialect.__slang_layer__(:add) == :core
      assert Dialect.__slang_metadata__("add") == %{suite: :sluice, opcode: 1}
      assert Dialect.__slang_metadata__("box_int") == %{wrapped: true}
      assert Slang.load(ctx, Dialect) |> MLIR.LogicalResult.success?()
      assert Context.terminator?(ctx, "sluice_slang.halt")

      frontend_ir =
        "module {\n  func.func @with_frontend(%x: i64, %code: i32) {\n    %0 = \"sluice_slang.box_int\"(%x) : (i64) -> i64\n    %w = \"sluice_slang.widen\"(%code) : (i32) -> i64\n    %1 = \"sluice_slang.const_int\"() {value = 5 : i64} : () -> i64\n    %2 = \"sluice_slang.add\"(%0, %w) : (i64, i64) -> i64\n    %3 = \"sluice_slang.add\"(%2, %1) : (i64, i64) -> i64\n    \"sluice_slang.halt\"(%code) : (i32) -> ()\n  }\n}\n"

      Module.with_module(ctx, frontend_ir, fn mod ->
        assert MLIR.verify!(mod)
        assert ^mod = Slang.verify_layers!(mod, Dialect, allow: [:frontend, :core, :exit])
        assert :ok = Slang.verify_layers(mod, Dialect, [:frontend, :core, :exit])

        assert {:error, [%{operation: "sluice_slang.box_int", layer: :frontend}]} =
                 Slang.verify_layers(mod, Dialect, allow: [:core, :exit])

        assert {:error, [%{operation: "sluice_slang.halt", layer: :exit}]} =
                 Slang.verify_layers(mod, Dialect, forbid: [:exit])

        assert_raise ArgumentError, ~r/layer verification failed/, fn ->
          Slang.verify_layers!(mod, Dialect, allow: [:core, :exit])
        end
      end)
    end
  end

  describe "8. Dialect-free @symbol Safepoint, PDL specialization, and Module.import_symbols!" do
    test "Safepoint supports @callee symbol matchers and constant operand poll cost", %{ctx: ctx} do
      ir =
        "module {\n  func.func private @beam.poll(i64) -> i1\n  func.func private @beam.suspend(i32) -> ()\n\n  func.func @leaf_fn(%state: i64, %frame: !llvm.ptr, %x: i64) -> i64 {\n    %c6 = arith.constant 6 : i64\n    %stop = func.call @beam.poll(%c6) : (i64) -> i1\n    %dbl = arith.addi %x, %x : i64\n    cf.cond_br %stop, ^yield, ^done\n  ^yield:\n    %zero = arith.constant 0 : i64\n    return %zero : i64\n  ^done:\n    return %dbl : i64\n  }\n\n  func.func @heavy_fn(%state: i64, %frame: !llvm.ptr, %x: i64) -> i64 {\n    %c40 = arith.constant 40 : i64\n    %stop = func.call @beam.poll(%c40) : (i64) -> i1\n    return %x : i64\n  }\n\n  func.func @suspending_fn(%state: i64, %frame: !llvm.ptr, %x: i64) -> i64 {\n    %c1 = arith.constant 1 : i32\n    func.call @beam.suspend(%c1) : (i32) -> ()\n    return %x : i64\n  }\n}\n"

      Module.with_module(ctx, ir, fn mod ->
        [leaf_fn, heavy_fn, suspending_fn] =
          mod
          |> Walker.find_ops("func.func")
          |> Enum.filter(fn op -> MLIR.Attribute.value(op["sym_name"]) in ["leaf_fn", "heavy_fn", "suspending_fn"] end)

        for {func, limit, suspend, expected} <- [
              {leaf_fn, 16, {"func.call", "@beam.suspend"}, true},
              {leaf_fn, 4, "@beam.suspend", false},
              {heavy_fn, 32, "@beam.suspend", false},
              {suspending_fn, 32, {"func.call", "@beam.suspend"}, false}
            ] do
          assert Safepoint.bounded_leaf?(func, poll_ops: ["@beam.poll"], suspend_ops: [suspend], max_work: limit) ==
                   expected
        end

        [poll] = Safepoint.resolve_ops(leaf_fn, "@beam.poll")

        [explicit, matched] = Safepoint.resolve_ops(leaf_fn, [poll, fn op -> Operation.name(op) == "arith.addi" end])
        assert MLIR.equal?(explicit, poll) and Operation.name(matched) == "arith.addi"
        analysis = Safepoint.analyze(leaf_fn, ["@beam.poll"], exclude_entry_args: true)
        assert length(analysis.safepoints) == 1

        bundled =
          Safepoint.bundle_leaf!(mod, "leaf_fn",
            poll_ops: ["@beam.poll"],
            suspend_ops: ["@beam.suspend"],
            state_arg: 0,
            frame_arg: 1,
            max_work: 16
          )

        assert bundled.bounded_leaf?
        assert bundled.leaf_symbol == "leaf_fn_leaf"
        assert MLIR.verify!(mod)
        leaf_text = MLIR.to_string(bundled.leaf_function)
        refute leaf_text =~ "@beam.poll"
        assert leaf_text =~ "arith.constant false"
      end)
    end

    test "PDL specialization + Module.import_symbols! + inline/sccp eliminates boxing and unused templates", %{
      ctx: ctx
    } do
      templates_ir =
        "module {\n  func.func private @beam.make_pair(%tag: i64, %val: i64) -> !llvm.struct<(i64, i64)> {\n    %u0 = llvm.mlir.undef : !llvm.struct<(i64, i64)>\n    %s0 = llvm.insertvalue %tag, %u0[0] : !llvm.struct<(i64, i64)>\n    %s1 = llvm.insertvalue %val, %s0[1] : !llvm.struct<(i64, i64)>\n    return %s1 : !llvm.struct<(i64, i64)>\n  }\n\n  func.func private @beam.box_int(%v: i64) -> !llvm.struct<(i64, i64)> {\n    %c1 = arith.constant 1 : i64\n    %res = func.call @beam.make_pair(%c1, %v) : (i64, i64) -> !llvm.struct<(i64, i64)>\n    return %res : !llvm.struct<(i64, i64)>\n  }\n\n  func.func private @beam.unbox_int(%t: !llvm.struct<(i64, i64)>) -> i64 {\n    %v = llvm.extractvalue %t[1] : !llvm.struct<(i64, i64)>\n    return %v : i64\n  }\n\n  func.func private @beam.unused_helper(%x: i64) -> i64 {\n    return %x : i64\n  }\n}\n"

      pdl_ir =
        "module {\n  pdl.pattern @unbox_box_int : benefit(2) {\n    %v = pdl.operand\n    %box_ty = pdl.type\n    %i64_ty = pdl.type : i64\n    %box_sym = pdl.attribute = @beam.box_int\n    %unbox_sym = pdl.attribute = @beam.unbox_int\n    %box = pdl.operation \"func.call\"(%v : !pdl.value) {\"callee\" = %box_sym} -> (%box_ty : !pdl.type)\n    %boxed = pdl.result 0 of %box\n    %unbox = pdl.operation \"func.call\"(%boxed : !pdl.value) {\"callee\" = %unbox_sym} -> (%i64_ty : !pdl.type)\n    pdl.rewrite %unbox {\n      pdl.replace %unbox with (%v : !pdl.value)\n    }\n  }\n}\n"

      user_ir =
        "module {\n  func.func private @beam.box_int(i64) -> !llvm.struct<(i64, i64)>\n  func.func private @beam.unbox_int(!llvm.struct<(i64, i64)>) -> i64\n\n  func.func @compute(%x: i64) -> !llvm.struct<(i64, i64)> {\n    %b0 = func.call @beam.box_int(%x) : (i64) -> !llvm.struct<(i64, i64)>\n    %u0 = func.call @beam.unbox_int(%b0) : (!llvm.struct<(i64, i64)>) -> i64\n    %sum = arith.addi %u0, %x : i64\n    %out = func.call @beam.box_int(%sum) : (i64) -> !llvm.struct<(i64, i64)>\n    return %out : !llvm.struct<(i64, i64)>\n  }\n}\n"

      Module.with_module(ctx, templates_ir, fn tmpl_mod ->
        Module.with_module(ctx, user_ir, fn user_mod ->
          Beaver.MLIR.Rewrite.apply_patterns!(user_mod, [{:pdl, pdl_ir}])
          after_pdl = MLIR.to_string(user_mod)
          refute after_pdl =~ "func.call @beam.unbox_int"
          Module.import_symbols!(user_mod, tmpl_mod, verify_type: true)
          imported_text = MLIR.to_string(user_mod)
          assert imported_text =~ "func.func private @beam.make_pair("
          refute imported_text =~ "@beam.unused_helper"
          assert MLIR.verify!(user_mod)

          ~w(inline sccp canonicalize cse symbol-dce)
          |> Enum.reduce(user_mod, &Composer.append(&2, &1))
          |> Composer.run!()

          final_text = MLIR.to_string(user_mod)
          assert_text(final_text, ["arith.addi", "llvm.insertvalue"], ["@beam.box_int", "@beam.make_pair", "func.call"])
        end)
      end)
    end
  end

  defp operation(name, bind, operands, results \\ [:i64], attrs \\ []),
    do: %{name: name, bind: bind, operands: operands, results: results, attributes: attrs}

  defp constant(bind, value), do: operation("arith.constant", bind, [], [:i64], value: {:i64, value})

  defp lower_llvm(mod) do
    ~w(convert-arith-to-llvm convert-cf-to-llvm convert-func-to-llvm reconcile-unrealized-casts)
    |> Enum.reduce(mod, &Composer.append(&2, &1))
    |> Composer.run!()
  end

  defp invoke_i64(jit, name, args) do
    result = Native.I64.make(0)
    ExecutionEngine.invoke!(jit, name, args, result)
    Native.to_term(result)
  end

  defp pointer(value), do: value |> Native.opaque_ptr() |> Native.to_term() |> Native.USize.make()

  defp assert_text(text, present, absent) do
    for part <- present, do: assert(text =~ part)
    for part <- absent, do: refute(text =~ part)
  end
end
