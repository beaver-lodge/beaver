defmodule Beaver.MLIR.Triton.LayoutAuditTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias Beaver.MLIR.Triton.LayoutAudit

  @fixture_path Path.expand("fixtures/triton/ttgir_convert_layout.mlir", __DIR__)
  @triton_enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil

  describe "parse_layout/1" do
    test "classifies the known Triton layout encodings" do
      assert %{kind: "blocked", params: "<{sizePerThread = [1, 1], threadsPerWarp = [32, 1]}>"} =
               LayoutAudit.parse_layout(
                 "#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1]}>"
               )

      assert %{kind: "shared", params: "<{vec = 1, perThread = 1}>"} =
               LayoutAudit.parse_layout("#ttg.shared<{vec = 1, perThread = 1}>")

      assert %{kind: "dot_operand", params: "<{opIdx = 0, parent = #ttg.blocked}>"} =
               LayoutAudit.parse_layout("#ttg.dot_operand<{opIdx = 0, parent = #ttg.blocked}>")
    end

    test "preserves unclassified text instead of dropping it" do
      assert %{kind: "unknown", params: nil, raw: "weird<thing>"} =
               LayoutAudit.parse_layout("weird<thing>")
    end
  end

  describe "extract_layout/1" do
    test "takes the layout encoding from a tensor type" do
      assert %{kind: "blocked", params: "<{sizePerThread = [1, 1]}>"} =
               LayoutAudit.extract_layout(
                 "tensor<64x2xi32, #ttg.blocked<{sizePerThread = [1, 1]}>>"
               )
    end

    test "classifies a tensor without an encoding as unknown" do
      assert %{kind: "unknown", raw: "tensor<64x2xi32>"} =
               LayoutAudit.extract_layout("tensor<64x2xi32>")
    end
  end

  describe "type_facts/1" do
    test "splits shape and element type" do
      assert %{shape: "64x2", element_type: "i32"} =
               LayoutAudit.type_facts("tensor<64x2xi32, #ttg.blocked>")

      assert %{shape: "64x2", element_type: "!tt.ptr<f32>"} =
               LayoutAudit.type_facts("tensor<64x2x!tt.ptr<f32>, #ttg.blocked1>")
    end
  end

  test "fixture is committed and mentions convert_layout" do
    fixture = File.read!(@fixture_path)
    assert fixture =~ "ttg.convert_layout"
    assert fixture =~ "#ttg.blocked"
  end

  @tag :triton
  @tag skip: !@triton_enabled
  test "audits a real TTGIR fixture end to end" do
    context = MLIR.Context.create(all_dialects: false)
    on_exit(fn -> MLIR.Context.destroy(context) end)
    Beaver.Triton.register(context)

    module =
      MLIR.Module.create!(File.read!(@fixture_path), ctx: context)

    on_exit(fn -> MLIR.Module.destroy(module) end)

    audit = LayoutAudit.audit(module)

    assert audit.operation_count == 4

    assert Enum.map(audit.convert_layouts, & &1.target_layout.kind) ==
             ["blocked", "blocked", "blocked", "blocked"]

    assert Enum.all?(audit.convert_layouts, &(is_binary(&1.location) and &1.location != ""))

    assert Enum.all?(audit.convert_layouts, fn conversion ->
             conversion.source_facts.shape == conversion.target_facts.shape
           end)
  end
end
