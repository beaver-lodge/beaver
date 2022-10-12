defmodule Beaver.MLIR.CAPI.CodeGen do
  @moduledoc false
  alias Kinda.CodeGen.{Type, NIF, Function}
  use Kinda.CodeGen

  @impl true
  def type_gen(_root_module, "?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void" = type) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.String.Callback}}
  end

  def type_gen(_root_module, "?fn(?*anyopaque) callconv(.C) void" = type) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.DiagnosticHandlerDeleteUserData}}
  end

  def type_gen(
        _root_module,
        "?fn(?*anyopaque, isize, c.struct_MlirAffineMap) callconv(.C) void" = type
      ) do
    {:ok,
     %Type{zig_t: type, module_name: Beaver.MLIR.AffineMapCompressUnusedSymbolsPopulateResult}}
  end

  def type_gen(
        _root_module,
        "?fn(c.struct_MlirDiagnostic, ?*anyopaque) callconv(.C) c.struct_MlirLogicalResult" = type
      ) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.DiagnosticHandler}}
  end

  def type_gen(
        _root_module,
        "?fn(?*anyopaque) callconv(.C) ?*anyopaque" = type
      ) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.ExternalPassConstruct}}
  end

  def type_gen(
        _root_module,
        "?fn(c.struct_MlirContext, ?*anyopaque) callconv(.C) c.struct_MlirLogicalResult" = type
      ) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.ExternalPassInitialize}}
  end

  def type_gen(
        _root_module,
        "?fn(c.struct_MlirOperation, c.struct_MlirExternalPass, ?*anyopaque) callconv(.C) void" =
          type
      ) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.ExternalPassRun}}
  end

  def type_gen(
        _root_module,
        "?fn(c.struct_MlirOperation, bool, ?*anyopaque) callconv(.C) void" = type
      ) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.SymbolTableWalkSymbolTablesCallback}}
  end

  def type_gen(
        _root_module,
        "?fn(isize, [*c]c.struct_MlirType, ?*anyopaque) callconv(.C) void" = type
      ) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.TypesCallback}}
  end

  def type_gen(_root_module, "c.struct_MlirPass" = type) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.Pass}}
  end

  def type_gen(_root_module, "c.struct_MlirValue" = type) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.Value}}
  end

  def type_gen(_root_module, "c.struct_MlirModule" = type) do
    {:ok, %Type{zig_t: type, module_name: Beaver.MLIR.Module}}
  end

  def type_gen(root_module, type) do
    Type.default(root_module, type) |> rewrite_module_name
  end

  defp rewrite_module_name({:ok, %{module_name: module_name} = type}) do
    new_name =
      if(
        module_name in [
          Beaver.MLIR.CAPI.OpaquePtr,
          Beaver.MLIR.CAPI.OpaqueArray,
          Beaver.MLIR.CAPI.Bool,
          Beaver.MLIR.CAPI.CInt,
          Beaver.MLIR.CAPI.CUInt,
          Beaver.MLIR.CAPI.F32,
          Beaver.MLIR.CAPI.F64,
          Beaver.MLIR.CAPI.I16,
          Beaver.MLIR.CAPI.I32,
          Beaver.MLIR.CAPI.I64,
          Beaver.MLIR.CAPI.I8,
          Beaver.MLIR.CAPI.ISize,
          Beaver.MLIR.CAPI.U16,
          Beaver.MLIR.CAPI.U32,
          Beaver.MLIR.CAPI.U64,
          Beaver.MLIR.CAPI.U8,
          Beaver.MLIR.CAPI.USize
        ]
      ) do
        base = module_name |> Module.split() |> List.last()
        Module.concat(Beaver.Native, base)
      else
        module_name
      end

    {:ok,
     %{
       type
       | module_name: new_name,
         kind_functions: memref_kind_functions()
     }}
  end

  defp memref_kind_functions() do
    [
      make: 5,
      aligned: 1,
      allocated: 1
    ]
  end

  @impl true
  def nif_gen(
        %Function{
          name: "mlirPassManagerRun"
        } = f
      ) do
    %{NIF.from_function(f) | dirty: :cpu}
  end

  def nif_gen(f) do
    NIF.from_function(f)
  end

  @impl true
  def func_filter(fns) do
    fns
    |> Enum.filter(fn x -> String.contains?(x, "mlir") || String.contains?(x, "beaver") end)
    |> Enum.filter(fn x -> String.contains?(x, "pub extern fn") end)
  end

  @impl true
  def kinds() do
    mem_ref_descriptor_kinds =
      for rank <- [
            DescriptorUnranked,
            Descriptor1D,
            Descriptor2D,
            Descriptor3D,
            Descriptor4D,
            Descriptor5D,
            Descriptor6D,
            Descriptor7D,
            Descriptor8D,
            Descriptor9D
          ],
          t <- [Complex.F32, U8, U16, U32, I8, I16, I32, I64, F32, F64] do
        %Kinda.CodeGen.Type{
          module_name: Module.concat([Beaver.Native, t, MemRef, rank]),
          kind_functions: memref_kind_functions()
        }
      end

    [
      %Kinda.CodeGen.Type{
        module_name: Beaver.Native.PtrOwner
      },
      %Kinda.CodeGen.Type{
        module_name: Beaver.Native.Complex.F32,
        kind_functions: memref_kind_functions()
      }
    ] ++ mem_ref_descriptor_kinds
  end
end
