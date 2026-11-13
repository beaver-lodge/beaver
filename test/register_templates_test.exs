defmodule Beaver.RegisterTemplatesTest do
  use Beaver.Case, async: true
  alias Beaver.{Composer, MLIR, Native}
  alias Beaver.MLIR.{ExecutionEngine, IRBatch, Module, SymbolTable}
  import IRBatch.Expressions

  defmodule Runtime do
    def functions, do: []
    def value_symbol(_), do: raise("no C bindings configured")
    def raw_symbol(_), do: raise("no C bindings configured")
    def value_parameters(_), do: raise("no C bindings configured")
  end

  defmodule Templates do
    @config [term: "!llvm.struct<(i64, i64, i64)>"]
    @config @config ++ [state: "!llvm.struct<(ptr, ptr, ptr, i64, i64, i64, i64, i64, ptr, i64, ptr, i64)>"]
    @config @config ++ [arithmetic: [{"band", "andi"}, {"mul", "muli"}, {"sub", "subi"}, {"add", "addi"}]]
    @config @config ++ [comparisons: [{"eq", "eq"}], heap_shapes: ["vector"], prefix: "vm.", runtime_prefix: "host"]
    @config @config ++ [runtime: Runtime, tag_mask: 65_535, cursor_shift: 16, quantum: 32]
    @config @config ++ [tags: %{empty: 16, integer: 17, boolean: 18, nil: 19, float: 20, heap: 21, cursor: 22}]
    use Beaver.MLIR.Template.Registers, @config ++ [services: %{binary_view: 41, checkpoint: 42}]
  end

  defmodule Patterns do
    use Beaver.MLIR.Template.Registers.Patterns,
      templates: Templates,
      runtime: Runtime,
      prefix: "vm.",
      tags: %{integer: 17, boolean: 18, nil: 19},
      policy: [layers: [:core], specialize_layers: [:core], import_layer: :core, prefix: "vm.", reserved: []]
  end

  test "configured providers generate verified libraries without Sluice names", %{ctx: ctx} do
    source = Templates.inline_source()
    refute source =~ "beam."
    refute source =~ "btt"
    refute source =~ "host_shape_tuple"
    assert source =~ "host_shape_vector"
    assert source =~ "42 : i64"
    assert source =~ "41 : i64"
    assert source =~ "32 : i64"

    Templates.with_inline_module(ctx, fn module ->
      assert MLIR.verify!(module)
      assert Map.has_key?(SymbolTable.function_signatures(module), "vm.host_helper")
    end)

    Templates.with_runtime_module(ctx, fn module ->
      assert MLIR.verify!(module)
      assert Map.keys(SymbolTable.function_signatures(module)) |> Enum.sort() == ~w(vm.host_cold vm.host_helper)
    end)

    rules = Patterns.core_pdl_source()
    assert rules =~ "@vm.int_add_i64"
    refute rules =~ "@vm.int_band_i64"
  end

  test "custom tags and cursor bits agree before and after PDL specialization", %{ctx: ctx} do
    t = Templates.term_type()
    integer = call("vm.box_int", [:x], t)
    nil_value = call("vm.box_nil", [], t)
    truth = call("vm.truth", [nil_value], t)
    binary = call("vm.box", [int(21), :x, int(0)], t)
    cursor = call("vm.binary_cursor", [binary, int(3)], t)
    difference = call("vm.scalar_sub", [integer, call("vm.box_int", [int(7)], t)], t)

    cases = [
      {"term_tag", integer, 17},
      {"term_tag", nil_value, 19},
      {"term_tag", truth, 18},
      {"term_bits", truth, 0},
      {"term_word0", cursor, 196_630},
      {"term_bits", difference, 35}
    ]

    for {accessor, value, expected} <- cases, specialize? <- [false, true] do
      spec = function("run", [x: :i64], :i64, call("vm." <> accessor, [value], :i64))
      spec = Map.merge(spec, %{visibility: :public, attributes: ["llvm.emit_c_interface": :unit]})

      Module.with_module(IRBatch.build_module!(ctx, [spec], verify: false), fn module ->
        Templates.ensure_declarations!(module)
        if specialize?, do: Patterns.specialize!(module)
        Templates.import_inline!(module)

        passes =
          ~w(inline canonicalize symbol-dce convert-arith-to-llvm convert-cf-to-llvm convert-func-to-llvm reconcile-unrealized-casts)

        Enum.reduce(passes, module, &Composer.append(&2, &1)) |> Composer.run!() |> MLIR.verify!()

        Native.with_resource(ExecutionEngine.create!(module), &ExecutionEngine.destroy/1, fn jit ->
          result = Native.I64.make(0)
          ExecutionEngine.invoke!(jit, "run", [Native.I64.make(42)], result)
          assert Native.to_term(result) == expected
        end)
      end)
    end
  end
end
