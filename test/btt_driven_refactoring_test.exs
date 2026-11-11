defmodule Beaver.BttDrivenRefactoringTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR

  alias Beaver.MLIR.{
    AnalysisSession,
    CAPI,
    ExecutionEngine,
    IRBatch,
    Safepoint,
    SymbolTable,
    Type,
    TypeConverter
  }

  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Target.LLVMIR
  alias Beaver.Pattern.Native.Program

  describe "1. Callback-free Rewrite Engine: Extended Pattern.Native.Program & PDLInterp" do
    test "Program :replace_op rewrites operations, selects operands, copies/sets attrs, and converts types with 0 callbacks",
         %{ctx: ctx} do
      MLIR.Context.allow_unregistered_dialects(ctx)

      swap_sub =
        Program.new("custom.rsub",
          operands: 2,
          results: 1,
          rewrite:
            {:replace_op, "arith.subi",
             operands: [1, 0], copy_attributes: [:tag], attributes: [rewritten: "true"]}
        )

      func_convert =
        Program.new("custom.func",
          rewrite:
            {:replace_op, "func.func",
             operands: [],
             result_types: [],
             copy_attributes: [:sym_name],
             regions: :move,
             convert_function_type: true}
        )

      ret_convert =
        Program.new("custom.return", rewrite: {:replace_op, "func.return", result_types: []})

      plan =
        Plan.new(mode: :full)
        |> Plan.add_legal_dialect("builtin")
        |> Plan.add_legal_dialect("func")
        |> Plan.add_legal_dialect("arith")
        |> Plan.add_illegal_dialect("custom")
        |> Plan.add_conversion_map(["i32"], "i64", recursive: true)
        |> Plan.add_structural_type_conversions([:func])
        |> Plan.add_pattern(swap_sub, version: "1.0")
        |> Plan.add_pattern(func_convert, version: "1.0")
        |> Plan.add_pattern(ret_convert, version: "1.0")

      ir = ~S"""
      module {
        "custom.func"() ({
        ^bb0(%arg0: i32, %arg1: i32):
          %0 = "custom.rsub"(%arg0, %arg1) {tag = "keep_me"} : (i32, i32) -> i32
          "custom.return"(%0) : (i32) -> ()
        }) {sym_name = "compute", function_type = (i32, i32) -> i32} : () -> ()
      }
      """

      Beaver.Native.with_resource(
        MLIR.Module.create!(ir, ctx: ctx),
        &MLIR.Module.destroy/1,
        fn module ->
          assert {{:ok, ^module, []}, receipt} = Plan.profile(plan, module)
          assert receipt["beam"]["callback_count"] == 0
          assert MLIR.verify!(module)
          ir_text = MLIR.to_string(module)
          assert ir_text =~ "func.func @compute(%arg0: i64, %arg1: i64) -> i64"
          assert ir_text =~ "arith.subi %arg1, %arg0"
          assert ir_text =~ ~s(tag = "keep_me") and ir_text =~ "rewritten = true"
          refute ir_text =~ "custom."
        end
      )
    end

    test "PDL patterns compile to PDLInterp and compose inside RewritePatternSet and Conversion.Plan across contexts" do
      pdl_source = ~S"""
      module {
        pdl.pattern @rewrite_custom_mul : benefit(1) {
          %t = pdl.type
          %lhs = pdl.operand : %t
          %rhs = pdl.operand : %t
          %op = pdl.operation "pdl_test.mul"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%t : !pdl.type)
          pdl.rewrite %op {
            %new_op = pdl.operation "arith.muli"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%t : !pdl.type)
            pdl.replace %op with %new_op
          }
        }
      }
      """

      plan =
        Plan.new(mode: :full)
        |> Plan.add_legal_dialect("builtin")
        |> Plan.add_legal_dialect("func")
        |> Plan.add_legal_dialect("arith")
        |> Plan.add_illegal_dialect("pdl_test")
        |> Plan.add_pdl_patterns(pdl_source, version: "1.0")

      ir = ~S"""
      module {
        func.func @mul(%a: i32, %b: i32) -> i32 {
          %0 = "pdl_test.mul"(%a, %b) : (i32, i32) -> i32
          return %0 : i32
        }
      }
      """

      for _run <- 1..2 do
        Beaver.Native.with_resource(
          MLIR.Context.create(),
          &MLIR.Context.destroy/1,
          fn fresh_ctx ->
            MLIR.Context.allow_unregistered_dialects(fresh_ctx)

            Beaver.Native.with_resource(
              MLIR.Module.create!(ir, ctx: fresh_ctx),
              &MLIR.Module.destroy/1,
              fn module ->
                assert {{:ok, ^module, []}, receipt} = Plan.profile(plan, module)
                assert receipt["beam"]["callback_count"] == 0
                assert MLIR.to_string(module) =~ "arith.muli"
                refute MLIR.to_string(module) =~ "pdl_test.mul"
              end
            )
          end
        )
      end
    end
  end

  describe "2. Conversion.Plan Structural Type Conversions, Recursive TypeConverter, SymbolTable & RewriterBase" do
    test "TypeConverter recursive conversion map rewrites nested composite types natively", %{
      ctx: ctx
    } do
      TypeConverter.with(fn converter ->
        TypeConverter.add_conversion_map(converter, [Type.i32(ctx: ctx)], Type.i64(ctx: ctx),
          recursive: true
        )

        {:ok, converted_struct} =
          TypeConverter.convert(
            converter,
            Type.get("!llvm.struct<(i32, !llvm.array<2 x i32>)>", ctx: ctx)
          )

        assert MLIR.to_string(converted_struct) == "!llvm.struct<(i64, array<2 x i64>)>"

        {:ok, converted_fn} =
          TypeConverter.convert(
            converter,
            Type.get("(i32, memref<4xi32>) -> (i32, vector<2xi32>)", ctx: ctx)
          )

        assert MLIR.to_string(converted_fn) == "(i64, memref<4xi64>) -> (i64, vector<2xi64>)"
      end)
    end

    test "Plan.add_structural_type_conversions converts func, cf, and scf signatures and control flow with 0 callbacks",
         %{ctx: ctx} do
      MLIR.Context.allow_unregistered_dialects(ctx)

      plan =
        Plan.new(mode: :full)
        |> Plan.add_legal_dialect("builtin")
        |> Plan.add_legal_dialect("arith")
        |> Plan.add_illegal_op("custom.Wrap")
        |> Plan.add_conversion_map(["i32"], "i64", recursive: true)
        |> Plan.add_structural_type_conversions([:func, :cf, :scf])
        |> Plan.add_pattern(Program.new("custom.Wrap", rewrite: {:operand, 0}), version: "1.0")

      ir = ~S"""
      module {
        func.func private @helper(i32) -> i32
        func.func @branch_and_if(%cond: i1, %x: i32) -> i32 {
          %0 = scf.if %cond -> (i32) {
            %wrapped = "custom.Wrap"(%x) : (i32) -> i32
            scf.yield %wrapped : i32
          } else {
            %called = func.call @helper(%x) : (i32) -> i32
            scf.yield %called : i32
          }
          cf.br ^bb1(%0 : i32)
        ^bb1(%merged: i32):
          return %merged : i32
        }
      }
      """

      Beaver.Native.with_resource(
        MLIR.Module.create!(ir, ctx: ctx),
        &MLIR.Module.destroy/1,
        fn module ->
          assert {{:ok, ^module, []}, receipt} = Plan.profile(plan, module)
          assert receipt["beam"]["callback_count"] == 0
          assert MLIR.verify!(module)
          rendered = MLIR.to_string(module)
          assert rendered =~ "func.func private @helper(i64) -> i64"
          assert rendered =~ "func.func @branch_and_if(%arg0: i1, %arg1: i64) -> i64"
          assert rendered =~ "scf.if %arg0 -> (i64)" and rendered =~ "cf.br ^bb1(%0 : i64)"
          refute rendered =~ "custom.Wrap"
        end
      )
    end

    test "SymbolTable.ensure_declaration! idempotently inserts external function declarations", %{
      ctx: ctx
    } do
      Beaver.Native.with_resource(
        MLIR.Module.create!("module {}", ctx: ctx),
        &MLIR.Module.destroy/1,
        fn module ->
          decl1 = SymbolTable.ensure_declaration!(module, "@btt_ext", "(i64) -> i32")
          decl2 = SymbolTable.ensure_declaration!(module, "btt_ext", "(i64) -> i32")
          assert MLIR.equal?(decl1, decl2)
          assert MLIR.to_string(module) =~ "func.func private @btt_ext(i64) -> i32"
        end
      )
    end
  end

  describe "3. BeaverLLVMIR: Host AOT Object Compiler & Per-Function Concurrent Sharding" do
    test "queries host target and compiles MLIR module directly to native object code and assembly",
         %{ctx: ctx} do
      target = LLVMIR.host_target()
      assert is_binary(target.triple) and byte_size(target.triple) > 0 and is_binary(target.cpu)

      ir = ~S"""
      module {
        llvm.func @add_i64(%a: i64, %b: i64) -> i64 {
          %sum = llvm.add %a, %b : i64
          llvm.return %sum : i64
        }
      }
      """

      Beaver.Native.with_resource(
        MLIR.Module.create!(ir, ctx: ctx),
        &MLIR.Module.destroy/1,
        fn module ->
          assert {:ok, obj_bytes, metrics} =
                   LLVMIR.compile_object(module, opt_level: 2, return_metrics: true)

          assert byte_size(obj_bytes) > 64 and metrics.translate_ns > 0 and metrics.codegen_ns > 0
          assert {:ok, asm_text} = LLVMIR.compile_object(module, format: :assembly, opt_level: 1)
          assert asm_text =~ "add_i64"
        end
      )
    end

    @tag :tmp_dir
    test "compile_functions shards a multi-function module concurrently and caches .o artifacts by SHA-256",
         %{ctx: ctx, tmp_dir: tmp_dir} do
      ir = ~S"""
      module {
        llvm.mlir.global external @shared_bias(10 : i64) : i64
        llvm.func @double(%x: i64) -> i64 {
          %two = llvm.mlir.constant(2 : i64) : i64
          %r = llvm.mul %x, %two : i64
          llvm.return %r : i64
        }
        llvm.func @double_plus_one(%x: i64) -> i64 {
          %d = llvm.call @double(%x) : (i64) -> i64
          %one = llvm.mlir.constant(1 : i64) : i64
          %r = llvm.add %d, %one : i64
          llvm.return %r : i64
        }
      }
      """

      Beaver.Native.with_resource(
        MLIR.Module.create!(ir, ctx: ctx),
        &MLIR.Module.destroy/1,
        fn module ->
          shards1 = LLVMIR.compile_functions!(module, cache_dir: tmp_dir, opt_level: 2)
          assert Enum.map(shards1, & &1.name) == ["double", "double_plus_one"]
          assert Enum.all?(shards1, &(not &1.cached? and byte_size(&1.object) > 64))

          shards2 = LLVMIR.compile_functions!(module, cache_dir: tmp_dir, opt_level: 2)
          assert Enum.all?(shards2, & &1.cached?)
          assert Enum.map(shards1, & &1.digest) == Enum.map(shards2, & &1.digest)
        end
      )
    end
  end

  describe "4. IRBatch.build: Single-NIF Batch IR Construction" do
    test "builds a multi-op basic block with intra-batch SSA and block-arg references in one submission",
         %{ctx: ctx} do
      ir = ~S"""
      module {
        func.func @batch_target(%x: i32, %y: i32) -> i32 {
          return %x : i32
        }
      }
      """

      Beaver.Native.with_resource(
        MLIR.Module.create!(ir, ctx: ctx),
        &MLIR.Module.destroy/1,
        fn module ->
          func = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)

          block =
            func |> Beaver.Walker.regions() |> Enum.at(0) |> Beaver.Walker.blocks() |> Enum.at(0)

          [old_ret] = Enum.to_list(Beaver.Walker.operations(block))
          i32 = Type.i32(ctx: ctx)

          specs = [
            %{
              name: "arith.constant",
              results: [i32],
              attributes: [value: MLIR.Attribute.integer(i32, 10)]
            },
            %{name: "arith.addi", operands: [{:block_arg, 0}, {:block_arg, 1}], results: [i32]},
            %{name: "arith.muli", operands: [{:op, 1}, {:result, 0, 0}], results: [i32]},
            %{name: "func.return", operands: [{:op, 2}]}
          ]

          assert {ops, results} = IRBatch.build!({:before, old_ret}, specs, ctx: ctx)
          assert length(ops) == 4 and length(results) == 4
          CAPI.mlirOperationRemoveFromParent(old_ret)
          MLIR.Operation.destroy(old_ret)

          assert MLIR.verify!(module)
          rendered = MLIR.to_string(module)

          assert rendered =~ "arith.constant 10 : i32" and
                   rendered =~ "arith.addi %arg0, %arg1 : i32" and rendered =~ "arith.muli"
        end
      )
    end
  end

  describe "5. ExecutionEngine Symbol Rebinding & C ABI Aggregate Call Lowering" do
    test "lowers C ABI aggregate calls with stacksave, alloca, store, load, stackrestore, and symbol declaration",
         %{ctx: ctx} do
      MLIR.Context.allow_unregistered_dialects(ctx)

      plan =
        Plan.new(mode: :full)
        |> Plan.add_legal_dialect("builtin")
        |> Plan.add_legal_dialect("llvm")
        |> Plan.add_illegal_dialect("btt")
        |> Plan.add_c_abi_call_pattern("btt.value_add", "llvm.call",
          callee: "@btt_c_value_add",
          return_type: "i32",
          params: [{:in, "i64"}, {:aggregate, "i64"}, {:aggregate, "i64"}, {:out, "i64"}],
          version: "1.0"
        )

      ir = ~S"""
      module {
        llvm.func @caller(%rt: i64, %lhs: i64, %rhs: i64) -> i64 {
          %status, %out = "btt.value_add"(%rt, %lhs, %rhs) : (i64, i64, i64) -> (i32, i64)
          llvm.return %out : i64
        }
      }
      """

      Beaver.Native.with_resource(
        MLIR.Module.create!(ir, ctx: ctx),
        &MLIR.Module.destroy/1,
        fn module ->
          assert {{:ok, ^module, []}, receipt} = Plan.profile(plan, module)
          assert receipt["beam"]["callback_count"] == 0
          assert MLIR.verify!(module)
          ir_text = MLIR.to_string(module)

          assert ir_text =~
                   "llvm.func @btt_c_value_add(i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32"

          assert ir_text =~ "llvm.intr.stacksave" and ir_text =~ "llvm.alloca" and
                   ir_text =~ "llvm.store"

          assert ir_text =~ "llvm.call @btt_c_value_add" and ir_text =~ "llvm.load" and
                   ir_text =~ "llvm.intr.stackrestore"

          refute ir_text =~ "btt.value_add"
        end
      )
    end

    @tag :tmp_dir
    test "ExecutionEngine binds native symbols under custom aliased names without global collisions",
         %{ctx: ctx, tmp_dir: tmp_dir} do
      lib_ir = ~S"""
      module {
        llvm.func @native_impl_run(%x: i32) -> i32 {
          %ten = llvm.mlir.constant(10 : i32) : i32
          %res = llvm.add %x, %ten : i32
          llvm.return %res : i32
        }
      }
      """

      obj_path = Path.join(tmp_dir, "impl.o")
      dylib_path = Path.join(tmp_dir, "libimpl.dylib")

      Beaver.Native.with_resource(
        MLIR.Module.create!(lib_ir, ctx: ctx),
        &MLIR.Module.destroy/1,
        fn lib_module ->
          File.write!(obj_path, LLVMIR.compile_object!(lib_module))
        end
      )

      {_, 0} = System.cmd("cc", ["-shared", "-o", dylib_path, obj_path])

      caller_ir = ~S"""
      module {
        func.func private @aliased_kernel(i32) -> i32
        func.func @entry(%x: i32) -> i32 attributes {llvm.emit_c_interface} {
          %r = func.call @aliased_kernel(%x) : (i32) -> i32
          return %r : i32
        }
      }
      """

      caller_module =
        MLIR.Module.create!(caller_ir, ctx: ctx)
        |> Beaver.Composer.append("convert-func-to-llvm")
        |> Beaver.Composer.run!()

      Beaver.Native.with_resource(caller_module, &MLIR.Module.destroy/1, fn caller_module ->
        assert ExecutionEngine.lookup_library_symbol(dylib_path, "nonexistent_symbol") == :error

        jit =
          ExecutionEngine.create!(caller_module,
            symbols: %{"aliased_kernel" => {dylib_path, "native_impl_run"}}
          )

        Beaver.Native.with_resource(jit, &ExecutionEngine.destroy/1, fn jit ->
          arg = Beaver.Native.I32.make(32)
          ret = Beaver.Native.I32.make(0)
          ExecutionEngine.invoke!(jit, "entry", [arg], ret)
          assert Beaver.Native.to_term(ret) == 42
        end)
      end)
    end
  end

  describe "6. BeaverAnalysis: Native Safepoint Liveness, Slot Coloring & Resumable CFG Lowering" do
    test "computes live-across-safepoint SSA values, shares non-overlapping slots, and lowers resumable CFG",
         %{ctx: ctx} do
      MLIR.Context.allow_unregistered_dialects(ctx)

      ir = ~S"""
      module {
        func.func @coro(%state: i32, %input: i64) -> i32 {
          %c1 = arith.constant 1 : i64
          %v1 = arith.addi %input, %c1 : i64
          "coro.suspend"() : () -> ()
          %c2 = arith.constant 2 : i64
          %v2 = arith.muli %v1, %c2 : i64
          "coro.suspend"() : () -> ()
          %done = arith.trunci %v2 : i64 to i32
          return %done : i32
        }
      }
      """

      Beaver.Native.with_resource(
        MLIR.Module.create!(ir, ctx: ctx),
        &MLIR.Module.destroy/1,
        fn module ->
          func_op = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)
          analysis = Safepoint.analyze(func_op, ["coro.suspend"], exclude_entry_args: true)
          assert analysis.slot_count == 1
          assert [%{slots: [0]}, %{slots: [0]}] = analysis.safepoints

          AnalysisSession.with(module, fn session ->
            assert AnalysisSession.safepoints(session, func_op, ["coro.suspend"],
                     exclude_entry_args: true
                   ).slot_count == 1
          end)

          assert ^func_op =
                   Safepoint.lower_resumable!(func_op, safepoints: ["coro.suspend"], state_arg: 0)

          assert MLIR.verify!(module)
          lowered_ir = MLIR.to_string(module)

          assert lowered_ir =~ "cf.switch" and lowered_ir =~ "coro.save" and
                   lowered_ir =~ "coro.load"

          refute lowered_ir =~ "coro.suspend"
        end
      )
    end
  end
end
