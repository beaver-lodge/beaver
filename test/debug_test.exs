defmodule DebugTest do
  @moduledoc """
  Test the debug configurations of MLIR.
  """
  use Beaver.Case, async: true
  use Beaver
  doctest Beaver.MLIR.Debug

  alias Beaver.MLIR
  alias MLIR.Dialect.{Arith, Func}
  alias MLIR.Type
  require Func

  import Beaver.MLIR.Conversion
  import Beaver.MLIR.Transform

  defp add_module(ctx) do
    mlir ctx: ctx do
      module do
        Func.func add(
                    function_type: Type.function([Type.i32(), Type.i32()], [Type.i32()]),
                    sym_name: MLIR.Attribute.string("add")
                  ) do
          region do
            block _(a >>> Type.i32(), b >>> Type.i32()) do
              r = Arith.addi(a, b) >>> Type.i32()
              Func.return(r) >>> []
            end
          end
        end
      end
    end
  end

  defp to_llvm(module) do
    module
    |> Beaver.Composer.nested("func.func", "llvm-request-c-wrappers")
    |> canonicalize()
    |> convert_scf_to_cf()
    |> convert_to_llvm()
    |> reconcile_unrealized_casts()
    |> Beaver.Composer.run!()
  end

  test "printing with debug_info does not attach DI scopes by itself", %{ctx: ctx} do
    text = ctx |> add_module() |> to_llvm() |> MLIR.to_string(debug_info: true)

    # Locations point at this test file, but no subprogram scopes yet.
    assert text =~ "debug_test.exs"
    refute text =~ "llvm.di_subprogram"
    refute text =~ "loc(fused<#di_subprogram"
  end

  test "attach_llvm_scopes! adds DI subprogram scopes for every llvm.func", %{ctx: ctx} do
    text =
      ctx
      |> add_module()
      |> to_llvm()
      |> Beaver.MLIR.Debug.attach_llvm_scopes!()
      |> MLIR.to_string(debug_info: true)

    assert text =~ ~s{#llvm.di_file<"debug_test.exs" in}
    assert text =~ "#llvm.di_compile_unit<"
    assert text =~ "#llvm.di_subprogram<"

    n_funcs = length(Regex.scan(~r/llvm\.func @/, text))
    n_scopes = length(Regex.scan(~r/loc\(fused<#di_subprogram/, text))
    assert n_funcs > 0
    assert n_funcs == n_scopes
  end

  test "attach_llvm_scopes! is idempotent", %{ctx: ctx} do
    module = ctx |> add_module() |> to_llvm()

    once =
      module
      |> Beaver.MLIR.Debug.attach_llvm_scopes!()
      |> MLIR.to_string(debug_info: true)

    twice =
      module
      |> Beaver.MLIR.Debug.attach_llvm_scopes!()
      |> Beaver.MLIR.Debug.attach_llvm_scopes!()
      |> MLIR.to_string(debug_info: true)

    assert once == twice
  end

  test "LLVM IR translation emits DILocation metadata for Elixir lines", %{ctx: ctx} do
    module = ctx |> add_module() |> to_llvm() |> Beaver.MLIR.Debug.attach_llvm_scopes!()
    ir = Beaver.MLIR.Target.LLVMIR.translate!(module)
    assert ir =~ "!dbg"
    assert ir =~ "!DILocation("
    assert ir =~ ~s{!DIFile(filename: "debug_test.exs"}
  end
end
