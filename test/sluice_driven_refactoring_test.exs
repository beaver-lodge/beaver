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
    any_of([
      Beaver.MLIR.Type.i64(),
      Beaver.MLIR.Type.i32()
    ])
  end

  def_pinned_unary_op(:box_int)

  Beaver.Slang.defops(
    [
      {:add, [lhs: :i64, rhs: :i64], [sum: :i64], [], [layer: :core, metadata: %{opcode: 1}]},
      {:widen, [val: :i64_or_i32], [out: :i64], [], [layer: :core, metadata: %{opcode: 3}]},
      {:const_int, [], [out: :i64], [value: "#builtin.integer"],
       [layer: :core, metadata: %{opcode: 2}]},
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

  alias Beaver.Changeset
  alias Beaver.Composer
  alias Beaver.MLIR

  alias Beaver.MLIR.{
    Block,
    Context,
    ConversionPatternRewriter,
    ExecutionEngine,
    Identifier,
    IRBatch,
    Location,
    Operation,
    Region,
    RewriterBase,
    Safepoint,
    SymbolTable,
    Template,
    Type,
    TypeConverter
  }

  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Target.LLVMIR
  alias Beaver.Pattern.Native.Program
  alias Beaver.Slang
  alias Beaver.SluiceDrivenRefactoringTest.LayeredSlangDialect
  alias Beaver.Walker

  describe "1. CAPI wrappers and with_* resource helpers" do
    test "Context.with_context and Module.with_module manage lifecycle and options" do
      rendered =
        Context.with_context([allow_unregistered: true], fn ctx ->
          assert MLIR.Context.allow_unregistered_dialects?(ctx)

          MLIR.Module.with_module(ctx, fn empty_mod ->
            assert MLIR.verify!(empty_mod)
            body = MLIR.Module.body(empty_mod)
            assert Enum.to_list(Block.operations(body)) == []
          end)

          ir = ~S"""
          module {
            "unreg.marker"() {tag = "ok"} : () -> ()
          }
          """

          MLIR.Module.with_module(ctx, ir, fn mod ->
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

        named =
          Block.add_named_args!(b1, [lhs: Type.i32(ctx: ctx), rhs: Type.i64(ctx: ctx)],
            location: loc
          )

        assert Map.keys(named) |> Enum.sort() == [:lhs, :rhs]
        assert length(Block.arguments(b1)) == 3
      after
        Region.destroy(region)
      end

      id_a = Identifier.get("sym_name", ctx: ctx)
      id_b = Identifier.get("sym_name", ctx: ctx)
      id_c = Identifier.get("other_attr", ctx: ctx)

      assert Identifier.to_string(id_a) == "sym_name"
      assert Identifier.equal?(id_a, id_b)
      refute Identifier.equal?(id_a, id_c)

      MLIR.Context.allow_unregistered_dialects(ctx)

      ir = ~S"""
      module {
        func.func @demo(%x: i64) -> i64 {
          %0 = "demo.first"(%x) {a = 1 : i64, b = "two", c = true} : (i64) -> i64
          %1 = "demo.second"(%0) : (i64) -> i64
          return %1 : i64
        }
      }
      """

      MLIR.Module.with_module(ctx, ir, fn mod ->
        mod_op = Operation.from_module(mod)
        mod_body = MLIR.Module.body(mod_op)
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
        assert rendered1 =~ "a = 1 : i64"
        assert rendered1 =~ ~s(b = "two")
        refute rendered1 =~ "c = true"

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
        assert text =~ "99 : i64"
        assert text =~ ~s(tag = "lit")
        assert text =~ "marker"
        assert text =~ "enabled = true"
        assert text =~ "callee = @target_fn"
        assert text =~ "array<i64: 2, 4>"
      after
        Operation.destroy(op)
      end
    end

    test "RewriterBase emit, scf_if, pack_struct, unpack_struct, and SymbolTable.ensure_declaration!",
         %{ctx: ctx} do
      MLIR.Context.allow_unregistered_dialects(ctx)

      plan =
        Plan.new(mode: :partial)
        |> Plan.add_legal_dialects(["builtin", "func", "arith", "scf", "llvm"])
        |> Plan.add_illegal_op("demo.select_and_roundtrip")
        |> Plan.add_conversion_pattern("demo.select_and_roundtrip", fn op,
                                                                       [cond_v, a_v, b_v],
                                                                       rw ->
          helper_ref =
            SymbolTable.ensure_declaration!(
              op,
              "ext_helper",
              "(i64) -> i64",
              op: "func.func",
              visibility: "private"
            )

          assert MLIR.to_string(helper_ref) =~ "func.func private @ext_helper(i64) -> i64"

          assert_raise ArgumentError, ~r/already declared with incompatible function_type/, fn ->
            SymbolTable.ensure_declaration!(
              op,
              "ext_helper",
              "(i32) -> i32",
              op: "func.func",
              verify_type: true
            )
          end

          [selected] =
            ConversionPatternRewriter.scf_if(
              rw,
              op,
              cond_v,
              [:i64],
              fn rw, then_loc ->
                doubled =
                  ConversionPatternRewriter.emit(rw, then_loc, "arith.addi", [a_v, a_v], [:i64])

                [doubled]
              end,
              fn rw, else_loc ->
                c10 =
                  ConversionPatternRewriter.emit(
                    rw,
                    else_loc,
                    "arith.constant",
                    [value: {:i64, 10}],
                    [:i64]
                  )

                shifted =
                  ConversionPatternRewriter.emit(rw, else_loc, "arith.addi", [b_v, c10], [:i64])

                [shifted]
              end
            )

          tag = RewriterBase.emit(rw, op, "arith.constant", [value: {:i32, 7}], [:i32])
          struct_ty = "!llvm.struct<(i64, i32)>"
          packed = ConversionPatternRewriter.pack_struct(rw, op, struct_ty, [selected, tag])

          [unpacked_val, _unpacked_tag] =
            ConversionPatternRewriter.unpack_struct(rw, op, packed, 2)

          ConversionPatternRewriter.replace_op(rw, op, [unpacked_val])
          :ok
        end)

      ir = ~S"""
      module {
        func.func @run(%flag: i1, %a: i64, %b: i64) -> i64 {
          %r = "demo.select_and_roundtrip"(%flag, %a, %b) : (i1, i64, i64) -> i64
          return %r : i64
        }
      }
      """

      MLIR.Module.with_module(ctx, ir, fn mod ->
        assert {:ok, ^mod, []} = Plan.run(plan, mod)
        assert MLIR.verify!(mod)
        rendered = MLIR.to_string(mod)
        assert rendered =~ "func.func private @ext_helper(i64) -> i64"
        assert rendered =~ "scf.if"
        assert rendered =~ "llvm.insertvalue"
        assert rendered =~ "llvm.extractvalue"
        refute rendered =~ "demo.select_and_roundtrip"
      end)
    end
  end

  describe "3. Multi-block IRBatch, inline attributes, and MLIR.Template" do
    test "IRBatch.build_module! builds multi-block loop and switch functions that JIT and execute",
         %{ctx: ctx} do
      MLIR.Context.register_translations(ctx)

      mod =
        IRBatch.build_module!(
          ctx,
          [
            {:func, "sum_up_to", [n: :i64], [:i64],
             attributes: [{"llvm.emit_c_interface", :unit}],
             blocks: [
               entry:
                 {[],
                  [
                    %{
                      name: "arith.constant",
                      attributes: [value: {:i64, 0}],
                      results: [:i64],
                      bind: :zero
                    },
                    %{
                      name: "arith.constant",
                      attributes: [value: {:i64, 1}],
                      results: [:i64],
                      bind: :one
                    },
                    {:br, {:loop, [:zero, :zero]}}
                  ]},
               loop:
                 {[i: :i64, acc: :i64],
                  [
                    %{
                      name: "arith.cmpi",
                      operands: [:i, :n],
                      attributes: [predicate: {:i64, 2}],
                      results: [:i1],
                      bind: :keep_going
                    },
                    {:cond_br, :keep_going, {:body, []}, {:done, [:acc]}}
                  ]},
               body:
                 {[],
                  [
                    %{name: "arith.addi", operands: [:i, :one], results: [:i64], bind: :next_i},
                    %{
                      name: "arith.addi",
                      operands: [:acc, :next_i],
                      results: [:i64],
                      bind: :next_acc
                    },
                    {:br, {:loop, [:next_i, :next_acc]}}
                  ]},
               done:
                 {[result: :i64],
                  [
                    {:return, [:result]}
                  ]}
             ]},
            {:func, "dispatch_op", [tag: :i32, x: :i64, y: :i64], [:i64],
             attributes: [{"llvm.emit_c_interface", :unit}],
             blocks: [
               entry:
                 {[],
                  [
                    {:switch, {:i32, :tag}, {:fallback, [:x]},
                     [
                       {0, {:add_case, [:x, :y]}},
                       {1, {:mul_case, [:x, :y]}}
                     ]}
                  ]},
               add_case:
                 {[a: :i64, b: :i64],
                  [
                    %{name: "arith.addi", operands: [:a, :b], results: [:i64], bind: :sum},
                    {:return, [:sum]}
                  ]},
               mul_case:
                 {[a: :i64, b: :i64],
                  [
                    %{name: "arith.muli", operands: [:a, :b], results: [:i64], bind: :prod},
                    {:return, [:prod]}
                  ]},
               fallback:
                 {[v: :i64],
                  [
                    {:return, [:v]}
                  ]}
             ]}
          ]
        )

      assert MLIR.verify!(mod)

      lowered =
        mod
        |> Composer.append("convert-arith-to-llvm")
        |> Composer.append("convert-cf-to-llvm")
        |> Composer.append("convert-func-to-llvm")
        |> Composer.append("reconcile-unrealized-casts")
        |> Composer.run!()

      try do
        ExecutionEngine.with_jit(lowered, fn jit ->
          n_arg = Beaver.Native.I64.make(10)
          sum_ret = Beaver.Native.I64.make(0)
          ExecutionEngine.invoke!(jit, "sum_up_to", [n_arg], sum_ret)
          assert Beaver.Native.to_term(sum_ret) == 55

          for {tag, expected} <- [{0, 13}, {1, 42}, {9, 6}] do
            t_arg = Beaver.Native.I32.make(tag)
            x_arg = Beaver.Native.I64.make(6)
            y_arg = Beaver.Native.I64.make(7)
            out = Beaver.Native.I64.make(0)
            ExecutionEngine.invoke!(jit, "dispatch_op", [t_arg, x_arg, y_arg], out)
            assert Beaver.Native.to_term(out) == expected
          end
        end)
      after
        MLIR.Module.destroy(lowered)
      end
    end

    test "Template.render! and Template.module! validate tokens and escape strings", %{ctx: ctx} do
      rendered =
        Template.render!(
          """
          module {
            func.func {{fn_sym}}({{arg_ssa}}: {{ty}}) -> {{ty}} attributes {label = {{note}}} {
            {{entry_blk}}:
              {{c_ssa}} = arith.constant {{bias}} : {{ty}}
              {{r_ssa}} = arith.addi {{arg_ssa}}, {{c_ssa}} : {{ty}}
              return {{r_ssa}} : {{ty}}
            }
          }
          """,
          fn_sym: {:sym, "add_bias"},
          arg_ssa: {:ssa, "x"},
          c_ssa: {:ssa, "bias_v"},
          r_ssa: {:ssa, "res"},
          entry_blk: {:blk, "bb0"},
          ty: {:type, :i64},
          bias: {:int, 17},
          note: {:str, "line1\n\"quoted\""}
        )

      assert rendered =~ "func.func @add_bias(%x: i64)"
      assert rendered =~ ~s(label = "line1\\n\\"quoted\\"")

      mod =
        Template.module!(
          ctx,
          """
          module {
            func.func {{sym}}(%x: i64) -> i64 {
              return %x : i64
            }
          }
          """,
          sym: {:sym, "id_fn"}
        )

      try do
        assert MLIR.verify!(mod)
      after
        MLIR.Module.destroy(mod)
      end

      assert_raise ArgumentError, ~r/invalid control character in MLIR symbol/, fn ->
        Template.check_symbol!("bad\nsymbol")
      end

      assert_raise ArgumentError, ~r/invalid MLIR SSA/, fn ->
        Template.check_ssa!("x) -> () {")
      end

      assert_raise ArgumentError, ~r/invalid MLIR block/, fn ->
        Template.check_block!("bb0: return")
      end
    end
  end

  describe "4. Walker.find_ops and Safepoint frame materialization + leaf fast-path" do
    test "Walker.find_ops filters by name, set, and predicate with exclude_root", %{ctx: ctx} do
      ir = ~S"""
      module {
        func.func @f(%a: i64, %b: i64) -> i64 {
          %c = arith.constant 5 : i64
          %s = arith.addi %a, %b : i64
          %m = arith.muli %s, %c : i64
          return %m : i64
        }
      }
      """

      MLIR.Module.with_module(ctx, ir, fn mod ->
        assert length(Walker.find_ops(mod, "arith.addi")) == 1
        assert length(Walker.find_ops(mod, ["arith.addi", "arith.muli"])) == 2

        [func_op] = Walker.find_ops(mod, "func.func")

        refute Enum.any?(
                 Walker.find_ops(func_op, :all, exclude_root: true),
                 &(Operation.name(&1) == "func.func")
               )

        assert Enum.any?(
                 Walker.find_ops(func_op, :all, exclude_root: false),
                 &(Operation.name(&1) == "func.func")
               )

        arith_ops =
          Walker.find_ops(func_op, fn op ->
            String.starts_with?(Operation.name(op), "arith.")
          end)

        assert length(arith_ops) == 3
      end)
    end

    test "Safepoint.lower_resumable! materializes native LLVM GEP/store/load with allow_unregistered: false and executes across resumes" do
      Context.with_context([allow_unregistered: false], fn ctx ->
        MLIR.Context.register_translations(ctx)

        ir = ~S"""
        module {
          func.func private @coro_suspend_marker() -> ()
          func.func @step_machine(%state: i32, %frame: !llvm.ptr, %input: i64) -> i64 attributes {llvm.emit_c_interface} {
            %c10 = arith.constant 10 : i64
            %v1 = arith.addi %input, %c10 : i64
            %extra = arith.muli %input, %c10 : i64
            func.call @coro_suspend_marker() : () -> ()
            %c3 = arith.constant 3 : i64
            %v2 = arith.muli %v1, %c3 : i64
            %v2_extra = arith.addi %v2, %extra : i64
            func.call @coro_suspend_marker() : () -> ()
            %c5 = arith.constant 5 : i64
            %v3 = arith.subi %v2_extra, %c5 : i64
            return %v3 : i64
          }
        }
        """

        mod = MLIR.Module.create!(ir, ctx: ctx)

        try do
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
          assert lowered_text =~ "llvm.getelementptr"
          assert lowered_text =~ "llvm.store"
          assert lowered_text =~ "llvm.load"
          refute lowered_text =~ "coro.save"
          refute lowered_text =~ "coro.load"
          refute MLIR.to_string(step_fn) =~ "@coro_suspend_marker"
        after
          MLIR.Module.destroy(mod)
        end

        # Also verify i32 status protocol end-to-end under JIT execution across state 0 -> 1 -> 2 -> 0!
        jit_ir = ~S"""
        module {
          func.func private @yield_point() -> ()
          func.func @coro_exec(%state: i32, %frame: !llvm.ptr, %out: !llvm.ptr, %input: i64) -> i32 attributes {llvm.emit_c_interface} {
            %c10 = arith.constant 10 : i64
            %v1 = arith.addi %input, %c10 : i64
            func.call @yield_point() : () -> ()
            %c3 = arith.constant 3 : i64
            %v2 = arith.muli %v1, %c3 : i64
            func.call @yield_point() : () -> ()
            llvm.store %v2, %out : i64, !llvm.ptr
            %done = arith.constant 0 : i32
            return %done : i32
          }
        }
        """

        exec_mod = MLIR.Module.create!(jit_ir, ctx: ctx)

        try do
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

          llvm_mod =
            exec_mod
            |> Composer.append("convert-arith-to-llvm")
            |> Composer.append("convert-cf-to-llvm")
            |> Composer.append("convert-func-to-llvm")
            |> Composer.append("reconcile-unrealized-casts")
            |> Composer.run!()

          ExecutionEngine.with_jit(llvm_mod, fn jit ->
            frame_slot = Beaver.Native.I64.make(0)
            frame_addr = frame_slot |> Beaver.Native.opaque_ptr() |> Beaver.Native.to_term()
            frame_ptr = Beaver.Native.USize.make(frame_addr)

            out_slot = Beaver.Native.I64.make(0)
            out_addr = out_slot |> Beaver.Native.opaque_ptr() |> Beaver.Native.to_term()
            out_ptr = Beaver.Native.USize.make(out_addr)

            input = Beaver.Native.I64.make(4)
            status = Beaver.Native.I32.make(-1)

            # Step 0 -> suspends at state 1 (spilling v1 = 14 into frame_slot)
            ExecutionEngine.invoke!(
              jit,
              "coro_exec",
              [Beaver.Native.I32.make(0), frame_ptr, out_ptr, input],
              status
            )

            assert Beaver.Native.to_term(status) == 1
            assert Beaver.Native.to_term(frame_slot) == 14

            # Step 1 -> reloads v1 = 14, computes v2 = 42, suspends at state 2 (spilling v2 = 42)
            ExecutionEngine.invoke!(
              jit,
              "coro_exec",
              [Beaver.Native.I32.make(1), frame_ptr, out_ptr, Beaver.Native.I64.make(999)],
              status
            )

            assert Beaver.Native.to_term(status) == 2
            assert Beaver.Native.to_term(frame_slot) == 42

            # Step 2 -> reloads v2 = 42, stores 42 into out_ptr, returns 0
            ExecutionEngine.invoke!(
              jit,
              "coro_exec",
              [Beaver.Native.I32.make(2), frame_ptr, out_ptr, Beaver.Native.I64.make(999)],
              status
            )

            assert Beaver.Native.to_term(status) == 0
            assert Beaver.Native.to_term(out_slot) == 42
          end)
        after
          MLIR.Module.destroy(exec_mod)
        end
      end)
    end

    test "Safepoint.bounded_leaf? and Safepoint.bundle_leaf! clone acyclic leaf functions and fold poll",
         %{ctx: ctx} do
      MLIR.Context.allow_unregistered_dialects(ctx)

      ir = ~S"""
      module {
        func.func @fast_add(%a: i64, %b: i64) -> i64 {
          %poll = "guest.poll_budget"() : () -> i1
          %res = scf.if %poll -> (i64) {
            %zero = arith.constant 0 : i64
            scf.yield %zero : i64
          } else {
            %sum = arith.addi %a, %b : i64
            scf.yield %sum : i64
          }
          return %res : i64
        }

        func.func @looping(%n: i64) -> i64 {
          cf.br ^hdr(%n : i64)
        ^hdr(%x: i64):
          %zero = arith.constant 0 : i64
          %cond = arith.cmpi sgt, %x, %zero : i64
          cf.cond_br %cond, ^hdr(%zero : i64), ^exit(%x : i64)
        ^exit(%out: i64):
          return %out : i64
        }
      }
      """

      MLIR.Module.with_module(ctx, ir, fn mod ->
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
        assert leaf_text =~ "@fast_add_leaf"
        refute leaf_text =~ "guest.poll_budget"
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
      MLIR.Context.allow_unregistered_dialects(ctx)

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
             copy_attributes: {:except, [:target_sym, :drop_tag]},
             attributes: [callee: {:symbol_ref, :target_sym}]}
        )

      ret_prog =
        Program.new("guest.return",
          rewrite: {:replace_op, "func.return", result_types: []}
        )

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
      ir = ~S"""
      module {
        func.func private @helper(i64) -> i64
        "guest.func"() ({
        ^bb0(%x: i32):
          %y = "guest.call"(%x) {target_sym = "helper", drop_tag = 99 : i64, keep_note = "preserved"} : (i32) -> i32
          "guest.return"(%y) : (i32) -> ()
        }) {sym_name = "entry", guest_meta = "stripped"} : () -> ()
      }
      """

      MLIR.Module.with_module(ctx, ir, fn mod ->
        assert {{:ok, ^mod, []}, receipt} = Plan.profile(plan, mod)
        assert receipt["beam"]["callback_count"] == 0
        assert {:ok, ^mod, []} = Plan.finish(plan, mod)
        assert MLIR.verify!(mod)

        rendered = MLIR.to_string(mod)
        assert rendered =~ "func.func @entry(%arg0: i64) -> i64"
        assert rendered =~ "call @helper(%arg0)"
        assert rendered =~ ~s(keep_note = "preserved")
        refute rendered =~ "guest_meta"
        refute rendered =~ "drop_tag"
        refute rendered =~ "target_sym"
      end)
    end

    test "multi-root 4-arity conversion callback, Plan.finish residual cast detection, and c_abi_call verify_declaration",
         %{ctx: ctx} do
      MLIR.Context.allow_unregistered_dialects(ctx)

      plan =
        Plan.new(mode: :partial)
        |> Plan.add_legal_dialects(["builtin", "func", "arith"])
        |> Plan.add_illegal_ops(["my.double_a", "my.double_b"])
        |> Plan.add_conversion_map(["i32"], "i64")
        |> Plan.add_structural_type_conversions([:func])
        |> Plan.add_conversion_pattern(["my.double_a", "my.double_b"], fn op,
                                                                          [v],
                                                                          rw,
                                                                          converter ->
          [orig_res_ty] = Operation.result_types(op)
          target_ty = TypeConverter.convert!(converter, orig_res_ty)
          sum = ConversionPatternRewriter.emit(rw, op, "arith.addi", [v, v], [target_ty])
          ConversionPatternRewriter.replace_op(rw, op, [sum])
          :ok
        end)

      ir = ~S"""
      module {
        func.func @calc(%x: i32) -> i32 {
          %0 = "my.double_a"(%x) : (i32) -> i32
          %1 = "my.double_b"(%0) : (i32) -> i32
          return %1 : i32
        }
      }
      """

      MLIR.Module.with_module(ctx, ir, fn mod ->
        assert ^mod = Plan.finish!(plan, mod)
        assert MLIR.to_string(mod) =~ "func.func @calc(%arg0: i64) -> i64"
      end)

      residual_ir = ~S"""
      module {
        func.func @leftover(%x: i32) -> i64 {
          %c = builtin.unrealized_conversion_cast %x : i32 to i64
          return %c : i64
        }
      }
      """

      MLIR.Module.with_module(ctx, residual_ir, fn mod ->
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

      mismatch_ir = ~S"""
      module {
        llvm.func @runtime_fn(!llvm.ptr) -> i64
        llvm.func @entry(%x: i64) -> i32 {
          %s = "ext.op"(%x) : (i64) -> i32
          llvm.return %s : i32
        }
      }
      """

      MLIR.Module.with_module(ctx, mismatch_ir, fn mod ->
        assert {:error, _diagnostics} = Plan.run(abi_plan, mod)
      end)
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
      assert desc1 == desc2
      assert desc1 != desc3

      counter = :atomics.new(2, signed: false)
      cache_dir = Path.join(tmp_dir, "unit_cache")

      units = [
        %{
          id: "math_add",
          key: "v1_math_add",
          source: fn ->
            :atomics.add(counter, 1, 1)

            """
            module {
              llvm.func @unit_add_ten(%x: i32) -> i32 {
                %c = llvm.mlir.constant(10 : i32) : i32
                %r = llvm.add %x, %c : i32
                llvm.return %r : i32
              }
            }
            """
          end
        },
        %{
          id: "math_mul",
          key: "v1_math_mul",
          source: fn ->
            :atomics.add(counter, 2, 1)

            """
            module {
              llvm.func @unit_mul_three(%x: i32) -> i32 {
                %c = llvm.mlir.constant(3 : i32) : i32
                %r = llvm.mul %x, %c : i32
                llvm.return %r : i32
              }
            }
            """
          end
        }
      ]

      cold = LLVMIR.compile_units!(units, cache_dir: cache_dir, opt_level: 2)
      assert :atomics.get(counter, 1) == 1
      assert :atomics.get(counter, 2) == 1
      assert Enum.map(cold, & &1.cached?) == [false, false]
      assert Enum.all?(cold, &(File.regular?(&1.path) and byte_size(&1.object) > 64))
      assert File.regular?(Path.join([cache_dir, "objects", "v1_math_add", "manifest.json"]))

      # Warm hit: zero source callback invocations, zero MLIR context needed!
      warm = LLVMIR.compile_units!(units, cache_dir: cache_dir, opt_level: 2)
      assert :atomics.get(counter, 1) == 1
      assert :atomics.get(counter, 2) == 1
      assert Enum.map(warm, & &1.cached?) == [true, true]
      assert Enum.map(warm, & &1.sha256) == Enum.map(cold, & &1.sha256)

      # Corrupt cache recovery: delete only the second unit's .o file
      File.rm!(Enum.at(warm, 1).path)
      recovered = LLVMIR.compile_units!(units, cache_dir: cache_dir, opt_level: 2)
      assert :atomics.get(counter, 1) == 1
      assert :atomics.get(counter, 2) == 2
      assert Enum.map(recovered, & &1.cached?) == [true, false]

      # Link both unit objects into a shared library
      shared_path = Path.join(tmp_dir, "libmath_units" <> shared_ext)
      assert ^shared_path = LLVMIR.link_shared!(recovered, shared_path)
      assert File.regular?(shared_path)

      # Invoke via ExecutionEngine.with_jit using fallback symbol list
      caller_ir = ~S"""
      module {
        func.func private @ext_add(i32) -> i32
        func.func private @ext_mul(i32) -> i32
        func.func @pipeline(%x: i32) -> i32 attributes {llvm.emit_c_interface} {
          %a = func.call @ext_add(%x) : (i32) -> i32
          %b = func.call @ext_mul(%a) : (i32) -> i32
          return %b : i32
        }
      }
      """

      caller_mod =
        MLIR.Module.create!(caller_ir, ctx: ctx)
        |> Composer.append("convert-func-to-llvm")
        |> Composer.run!()

      try do
        result_val =
          ExecutionEngine.with_jit(
            caller_mod,
            [
              symbols: %{
                "ext_add" => {shared_path, ["nonexistent_add_sym", "unit_add_ten"]},
                "ext_mul" => {shared_path, ["missing_a", "missing_b", "unit_mul_three"]}
              }
            ],
            fn jit ->
              arg = Beaver.Native.I32.make(4)
              ret = Beaver.Native.I32.make(0)
              ExecutionEngine.invoke!(jit, "pipeline", [arg], ret)
              Beaver.Native.to_term(ret)
            end
          )

        assert result_val == 42
      after
        MLIR.Module.destroy(caller_mod)
      end
    end
  end

  describe "7. Beaver.Slang hygienic pins, table-driven defops, layers, metadata, and schema digest" do
    test "loads hygienic macro-pinned and table-driven Slang ops, exposes schema digest, and verifies layers",
         %{ctx: ctx} do
      digest = LayeredSlangDialect.__slang_schema_digest__()
      assert String.starts_with?(digest, "sha256:") and byte_size(digest) == 71
      assert Slang.schema_digest(LayeredSlangDialect) == digest

      assert LayeredSlangDialect.__slang_operations__() == [
               "sluice_slang.box_int",
               "sluice_slang.add",
               "sluice_slang.widen",
               "sluice_slang.const_int",
               "sluice_slang.halt"
             ]

      assert LayeredSlangDialect.__slang_operations__(:core) == [
               "sluice_slang.add",
               "sluice_slang.widen",
               "sluice_slang.const_int"
             ]

      assert LayeredSlangDialect.__slang_operations__(:frontend) == ["sluice_slang.box_int"]
      assert LayeredSlangDialect.__slang_operations__(:exit) == ["sluice_slang.halt"]

      assert LayeredSlangDialect.__slang_layer__("box_int") == :frontend
      assert LayeredSlangDialect.__slang_layer__(:add) == :core

      assert LayeredSlangDialect.__slang_metadata__("add") == %{suite: :sluice, opcode: 1}
      assert LayeredSlangDialect.__slang_metadata__("box_int") == %{wrapped: true}

      assert Slang.load(ctx, LayeredSlangDialect) |> MLIR.LogicalResult.success?()
      assert MLIR.Context.terminator?(ctx, "sluice_slang.halt")

      frontend_ir = ~S"""
      module {
        func.func @with_frontend(%x: i64, %code: i32) {
          %0 = "sluice_slang.box_int"(%x) : (i64) -> i64
          %w = "sluice_slang.widen"(%code) : (i32) -> i64
          %1 = "sluice_slang.const_int"() {value = 5 : i64} : () -> i64
          %2 = "sluice_slang.add"(%0, %w) : (i64, i64) -> i64
          %3 = "sluice_slang.add"(%2, %1) : (i64, i64) -> i64
          "sluice_slang.halt"(%code) : (i32) -> ()
        }
      }
      """

      MLIR.Module.with_module(ctx, frontend_ir, fn mod ->
        assert MLIR.verify!(mod)

        assert ^mod =
                 Slang.verify_layers!(mod, LayeredSlangDialect, allow: [:frontend, :core, :exit])

        assert :ok = Slang.verify_layers(mod, LayeredSlangDialect, [:frontend, :core, :exit])

        assert {:error, [%{operation: "sluice_slang.box_int", layer: :frontend}]} =
                 Slang.verify_layers(mod, LayeredSlangDialect, allow: [:core, :exit])

        assert {:error, [%{operation: "sluice_slang.halt", layer: :exit}]} =
                 Slang.verify_layers(mod, LayeredSlangDialect, forbid: [:exit])

        assert_raise ArgumentError, ~r/layer verification failed/, fn ->
          Slang.verify_layers!(mod, LayeredSlangDialect, allow: [:core, :exit])
        end
      end)
    end
  end
end
