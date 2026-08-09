defmodule DLTITest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.DLTI

  @moduletag :smoke

  test "builds and attaches a semantic data layout", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
        end
      end

    spec =
      DLTI.data_layout(
        endianness: :little,
        mangling_mode: "elf",
        ctx: ctx
      )

    assert to_string(spec) =~ ~s("dlti.endianness" = "little")
    assert to_string(spec) =~ ~s("dlti.mangling_mode" = "elf")

    assert ^module = DLTI.attach(module, spec)
    MLIR.verify!(module)
    assert to_string(module) =~ "dlti.dl_spec"
  end

  test "supports type keys and explicit attribute values", %{ctx: ctx} do
    entry =
      DLTI.entry(
        MLIR.Type.i32(),
        MLIR.Attribute.integer(MLIR.Type.i64(), 32),
        ctx: ctx
      )

    assert to_string(entry) == "#dlti.dl_entry<i32, 32 : i64>"
  end

  test "builds and attaches a target system spec", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
        end
      end

    target =
      DLTI.target_system_spec(
        [
          {"CPU", [{"dlti.L1_cache_size_in_bytes", 4096}]},
          {"GPU", [{"dlti.max_vector_op_width", 128}]}
        ],
        ctx: ctx
      )

    assert to_string(target) =~ ~s("CPU" = #dlti.target_device_spec<)
    assert to_string(target) =~ ~s("GPU" = #dlti.target_device_spec<)

    assert ^module = DLTI.attach_target_system(module, target)
    MLIR.verify!(module)
  end

  test "validates semantic options and duplicate keys", %{ctx: ctx} do
    assert_raise ArgumentError, ~r/endianness must be/, fn ->
      DLTI.data_layout(endianness: :middle, ctx: ctx)
    end

    assert_raise ArgumentError, ~r/duplicate DLTI entry key/, fn ->
      DLTI.spec([{"test.id", 1}, {"test.id", 2}], ctx: ctx)
    end
  end
end
