defmodule WasmTermABITest do
  use ExUnit.Case, async: true

  alias Beaver.Wasm.TermABI

  test "covers exactly the term intrinsics emitted by the conversion plan" do
    coverage = TermABI.coverage()
    assert coverage.missing == []
    assert coverage.extra == []

    assert length(TermABI.manifest()) ==
             length(Beaver.MLIR.Conversion.Ex.term_intrinsic_symbols()) -
               length(TermABI.native_only_symbols())
  end

  test "excludes native scheduler intrinsics from the wasm host boundary" do
    assert TermABI.native_only_symbols() == ["ex.term.process_wait", "ex.term.worker_run"]

    refute TermABI.entry("ex.term.process_wait")
    refute TermABI.entry("ex.term.worker_run")
  end

  test "declares the wasm import module and name" do
    list_cons = TermABI.entry("ex.term.list_cons")

    assert list_cons.import_module == "ex_term"
    assert list_cons.import_name == "ex.term.list_cons"
  end

  test "declares term signatures" do
    assert TermABI.entry("ex.term.list_cons").params == [:i64, :i64]
    assert TermABI.entry("ex.term.list_cons").result == :i64
    assert TermABI.entry("ex.term.send").params == [:i64, :i64]
    assert TermABI.entry("ex.term.runtime_create").params == []
    assert TermABI.entry("ex.term.runtime_enter").params == [:i64]
    assert TermABI.entry("ex.term.runtime_leave").params == []
    assert TermABI.entry("ex.term.runtime_destroy").params == [:i64]
    assert TermABI.entry("ex.term.export").params == [:i64, :i64]
    assert TermABI.entry("ex.term.import").params == [:i64, :i64]
    assert TermABI.entry("ex.term.handle_destroy").params == [:i64]
    assert TermABI.entry("ex.term.clock_tick").params == [:i64]
    assert TermABI.entry("ex.term.stream_filter").params == [:i64, :i64]
    assert TermABI.entry("ex.term.enumerable_reduce_c").params == [:i64, :i64, :i64, :i64]
  end

  test "declares the pointer and noreturn special cases" do
    assert TermABI.entry("ex.term.try_push").params == [:ptr]
    assert TermABI.entry("ex.term.throw").params == [:i64]
    assert TermABI.entry("ex.term.throw").result == :void
    assert TermABI.entry("ex.term.make_fun").params == [:i64, :i64, :i64, :i64, :i64, :i64]
  end

  test "renders a paste-ready markdown table" do
    rendered = TermABI.render()

    assert rendered =~ "## ex.term.* wasm host imports"
    assert rendered =~ "| `ex.term.list_cons` | `ex_term` | `(i64, i64) -> i64` |"
    assert rendered =~ "| `ex.term.throw` | `ex_term` | `(i64) -> noreturn` |"
  end
end
