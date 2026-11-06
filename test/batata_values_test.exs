defmodule BatataValuesTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Ex
  alias Beaver.MLIR.Conversion.Ex.CRuntime

  setup %{ctx: ctx} do
    Beaver.Slang.load(ctx, Beaver.MLIR.Dialect.Ex)
    :ok
  end

  test "every owned value type crosses calls, block arguments and structured loops", %{ctx: ctx} do
    for type <- Map.keys(CRuntime.value_types()) do
      module =
        parse!(ctx, """
        module {
          func.func @identity(%v: #{type}) -> #{type} { return %v : #{type} }
          func.func @flow(%v: #{type}, %c: i1, %lo: index, %hi: index, %step: index) -> #{type} {
            %loop = scf.for %i = %lo to %hi step %step iter_args(%acc = %v) -> (#{type}) {
              %next = scf.if %c -> (#{type}) { scf.yield %acc : #{type} } else { scf.yield %v : #{type} }
              scf.yield %next : #{type}
            }
            %while = scf.while (%acc = %loop) : (#{type}) -> #{type} {
              scf.condition(%c) %acc : #{type}
            } do {
            ^bb0(%next: #{type}):
              scf.yield %next : #{type}
            }
            cf.cond_br %c, ^taken(%while : #{type}), ^other(%v : #{type})
          ^taken(%x: #{type}):
            %y = func.call @identity(%x) : (#{type}) -> #{type}
            return %y : #{type}
          ^other(%z: #{type}):
            return %z : #{type}
          }
        }
        """)
        |> Ex.finish!(runtime_namespace: "typed_values")
        |> MLIR.verify!()

      refute MLIR.to_string(module) =~ "!ex."

      Enum.reduce(
        ~w(convert-scf-to-cf convert-arith-to-llvm convert-cf-to-llvm convert-func-to-llvm reconcile-unrealized-casts),
        module,
        &Beaver.Composer.append(&2, &1)
      )
      |> Beaver.Composer.run!()
      |> MLIR.verify!()
    end
  end

  test "Ex functions preserve owned terms without tagged-word conversion", %{ctx: ctx} do
    module =
      parse!(ctx, ~S"""
      module {
        "ex.func"() ({
        ^bb0(%term: !ex.term):
          "ex.return"(%term) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> ()
        }) {sym_name = "identity", sym_visibility = "private", test.marker = "preserved"} : () -> ()
        "ex.func"() ({
        ^bb0(%term: !ex.term):
          %v = "ex.call"(%term) {callee = "identity", arity = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>} : (!ex.term) -> !ex.term
          "ex.return"(%v) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> ()
        }) {sym_name = "caller"} : () -> ()
      }
      """)
      |> Ex.finish!(runtime_namespace: "typed_values")
      |> MLIR.verify!()

    assert MLIR.to_string(module) =~ "!llvm.struct<(i64, i64, i32, i32)>"
    assert MLIR.to_string(module) =~ "func.func private @identity"
    assert MLIR.to_string(module) =~ ~s(test.marker = "preserved")
    refute MLIR.to_string(module) =~ "!ex."
  end

  test "value calls can cross retained execution and runtime layers", %{ctx: ctx} do
    source = ~S"""
    func.func @create(%r: !llvm.ptr, %entry: !ex.entry, %request: i64, %size: i64) -> (i32, !ex.import_token) {
      %s, %p = "ex.value_process"(%r, %entry) : (!llvm.ptr, !ex.entry) -> (i32, !ex.process)
      %t, %token = "ex.value_binary_begin"(%r, %p, %request, %size) : (!llvm.ptr, !ex.process, i64, i64) -> (i32, !ex.import_token)
      return %t, %token : i32, !ex.import_token
    }
    """

    for layers <- [[:execution, :runtime], [:runtime, :execution]] do
      module = parse!(ctx, source)

      module =
        Enum.reduce(layers, module, fn layer, ir ->
          Beaver.MLIR.Conversion.Plan.run!(
            Ex.plan(layers: layer, runtime_namespace: "staged"),
            ir
          )
          |> MLIR.verify!()
        end)
        |> Ex.finish!(runtime_namespace: "staged")

      refute MLIR.to_string(module) =~ "!ex."
    end
  end

  test "native function signatures still convert internal handle block arguments", %{ctx: ctx} do
    for {function, branch, return} <- [
          {"func.func", "cf.br", "return"},
          {"llvm.func", "cf.br", "llvm.return"}
        ] do
      module =
        parse!(ctx, """
        #{function} @internal(%slot: !llvm.ptr) {
          %value = "ex.value_load"(%slot) : (!llvm.ptr) -> !ex.import_token
          #{branch} ^next(%value : !ex.import_token)
        ^next(%value2: !ex.import_token):
          "ex.value_store"(%value2, %slot) : (!ex.import_token, !llvm.ptr) -> ()
          #{return}
        }
        """)
        |> Ex.finish!(runtime_namespace: "internal_blocks")
        |> MLIR.verify!()

      refute MLIR.to_string(module) =~ "!ex."
    end
  end

  test "unsupported language operations roll back converted regions and function signatures", %{
    ctx: ctx
  } do
    for function <- [:func, :ex] do
      body = ~S"""
      %v = scf.if %c -> (!ex.term) { scf.yield %term : !ex.term } else { scf.yield %term : !ex.term }
      %unsupported = "ex.unbox"(%v) : (!ex.term) -> i64
      """

      source =
        if function == :func,
          do:
            "func.func @fail(%term: !ex.term, %c: i1) -> i64 { #{body} return %unsupported : i64 }",
          else: ~s["ex.func"() ({ ^bb0(%term: !ex.term, %c: i1): #{body}
            "ex.return"(%unsupported) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "fail"} : () -> ()]

      module = parse!(ctx, source)
      original = MLIR.to_string(module)

      assert {:error, %MLIR.Conversion.Error{}} =
               Beaver.MLIR.Conversion.Plan.run(Ex.plan(runtime_namespace: "rollback"), module)

      MLIR.verify!(module)
      assert MLIR.to_string(module) == original
    end
  end

  defp parse!(ctx, source), do: MLIR.Module.create!(source, ctx: ctx) |> MLIR.verify!()
end
