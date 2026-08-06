defmodule PtrTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  alias MLIR.Dialect.{Func, Ptr}
  require Func

  @moduletag :smoke

  @llvm_ptr_module """
  module {
    llvm.func @ptr_load_store(%ptr: !ptr.ptr<#llvm.address_space<0>>, %value: i64) -> i64 {
      ptr.store %value, %ptr : i64, !ptr.ptr<#llvm.address_space<0>>
      %loaded = ptr.load %ptr : !ptr.ptr<#llvm.address_space<0>> -> i64
      llvm.return %loaded : i64
    }

    llvm.func @_mlir_ciface_ptr_load_store(%ptr: !ptr.ptr<#llvm.address_space<0>>, %value: i64) -> i64 attributes {llvm.emit_c_interface} {
      %result = llvm.call @ptr_load_store(%ptr, %value) : (!ptr.ptr<#llvm.address_space<0>>, i64) -> i64
      llvm.return %result : i64
    }

    llvm.func @ptr_distance(%ptr: !ptr.ptr<#llvm.address_space<0>>, %offset: i64) -> i64 {
      %next = ptr.ptr_add %ptr, %offset : !ptr.ptr<#llvm.address_space<0>>, i64
      %distance = ptr.ptr_diff %next, %ptr : !ptr.ptr<#llvm.address_space<0>> -> i64
      llvm.return %distance : i64
    }

    llvm.func @_mlir_ciface_ptr_distance(%ptr: !ptr.ptr<#llvm.address_space<0>>, %offset: i64) -> i64 attributes {llvm.emit_c_interface} {
      %result = llvm.call @ptr_distance(%ptr, %offset) : (!ptr.ptr<#llvm.address_space<0>>, i64) -> i64
      llvm.return %result : i64
    }

    llvm.func @constant_distance() -> i64 {
      %null = ptr.constant #ptr.null : !ptr.ptr<#llvm.address_space<0>>
      %address = ptr.constant #ptr.address<0x10> : !ptr.ptr<#llvm.address_space<0>>
      %distance = ptr.ptr_diff %address, %null : !ptr.ptr<#llvm.address_space<0>> -> i64
      llvm.return %distance : i64
    }

    llvm.func @_mlir_ciface_constant_distance() -> i64 attributes {llvm.emit_c_interface} {
      %result = llvm.call @constant_distance() : () -> i64
      llvm.return %result : i64
    }

    llvm.func @vector_memory(
        %ptr: !ptr.ptr<#llvm.address_space<3>>,
        %ptrs: vector<4x!ptr.ptr<#llvm.address_space<3>>>,
        %mask: vector<4xi1>,
        %value: vector<4xf32>) -> vector<4xf32> {
      %loaded = ptr.masked_load %ptr, %mask, %value alignment = 4 : !ptr.ptr<#llvm.address_space<3>> -> vector<4xf32>
      ptr.masked_store %loaded, %ptr, %mask alignment = 4 : vector<4xf32>, !ptr.ptr<#llvm.address_space<3>>
      %gathered = ptr.gather %ptrs, %mask, %value alignment = 4 : vector<4x!ptr.ptr<#llvm.address_space<3>>> -> vector<4xf32>
      ptr.scatter %gathered, %ptrs, %mask alignment = 4 : vector<4xf32>, vector<4x!ptr.ptr<#llvm.address_space<3>>>
      llvm.return %gathered : vector<4xf32>
    }
  }
  """

  test "constructs Ptr types and typed constants", %{ctx: ctx} do
    assert Ptr.generic_space(ctx: ctx) |> MLIR.to_string() == "#ptr.generic_space"
    assert Ptr.llvm_address_space(3, ctx: ctx) |> MLIR.to_string() == "#llvm.address_space<3>"
    assert Ptr.type(ctx: ctx) |> MLIR.to_string() == "!ptr.ptr<#ptr.generic_space>"

    assert Ptr.type(memory_space: {:llvm, 3}, ctx: ctx) |> MLIR.to_string() ==
             "!ptr.ptr<#llvm.address_space<3>>"

    assert Ptr.null(ctx: ctx) |> MLIR.to_string() ==
             "#ptr.null : !ptr.ptr<#ptr.generic_space>"

    assert Ptr.address(4096, memory_space: {:llvm, 1}, ctx: ctx) |> MLIR.to_string() ==
             "#ptr.address<4096> : !ptr.ptr<#llvm.address_space<1>>"

    assert_raise ArgumentError, fn -> Ptr.address(-1, ctx: ctx) end
    assert_raise ArgumentError, fn -> Ptr.llvm_address_space(-1, ctx: ctx) end
  end

  test "constructs ptr.constant through the Beaver DSL", %{ctx: ctx} do
    ptr_type = Ptr.type()

    module =
      mlir ctx: ctx do
        module do
          Func.func constants(function_type: Type.function([], [ptr_type, ptr_type])) do
            region do
              block _() do
                null = Ptr.constant(value: Ptr.null(type: ptr_type)) >>> :infer
                address = Ptr.constant(value: Ptr.address(0x1000, type: ptr_type)) >>> :infer
                Func.return(null, address) >>> []
              end
            end
          end
        end
      end
      |> MLIR.verify!()

    ir = MLIR.to_string(module)
    assert ir =~ "ptr.constant #ptr.null"
    assert ir =~ "ptr.constant #ptr.address<4096>"
  end

  test "executes scalar Ptr operations and translates shaped operations", %{ctx: ctx} do
    MLIR.Context.register_translations(ctx)
    module = MLIR.Module.create!(@llvm_ptr_module, ctx: ctx) |> MLIR.verify!()
    engine = MLIR.ExecutionEngine.create!(module)

    try do
      storage = Beaver.Native.I64.make(1)
      address = storage |> Beaver.Native.opaque_ptr() |> Beaver.Native.to_term()
      pointer_argument = Beaver.Native.USize.make(address)
      value = Beaver.Native.I64.make(42)
      result = Beaver.Native.I64.make(0)

      MLIR.ExecutionEngine.invoke!(engine, "ptr_load_store", [pointer_argument, value], result)
      assert Beaver.Native.to_term(storage) == 42
      assert Beaver.Native.to_term(result) == 42

      offset = Beaver.Native.I64.make(24)
      distance = Beaver.Native.I64.make(0)
      MLIR.ExecutionEngine.invoke!(engine, "ptr_distance", [pointer_argument, offset], distance)
      assert Beaver.Native.to_term(distance) == 24

      constant_distance = Beaver.Native.I64.make(0)
      MLIR.ExecutionEngine.invoke!(engine, "constant_distance", [], constant_distance)
      assert Beaver.Native.to_term(constant_distance) == 16

      assert %Beaver.Native.OpaquePtr{} =
               MLIR.ExecutionEngine.lookup(engine, "vector_memory")
    after
      MLIR.ExecutionEngine.destroy(engine)
      MLIR.Module.destroy(module)
    end
  end
end
