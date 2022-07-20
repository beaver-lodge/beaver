defmodule Beaver.MLIR.CAPI.CodeGen do
  alias Fizz.CodeGen.Type

  def type_gen("?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void" = type) do
    {:ok,
     %Type{
       zig_t: type,
       module_name: :MlirStringCallback,
       delegates: [
         {Beaver.MLIR.StringRef.CallbackImpl, :create, 0}
       ]
     }}
  end

  def type_gen("?fn(?*anyopaque) callconv(.C) void" = type) do
    {:ok, %Type{zig_t: type, module_name: :DiagnosticHandlerDeleteUserData}}
  end

  def type_gen("?fn(?*anyopaque, isize, c.struct_MlirAffineMap) callconv(.C) void" = type) do
    {:ok, %Type{zig_t: type, module_name: :AffineMapCompressUnusedSymbolsPopulateResult}}
  end

  def type_gen(
        "?fn(c.struct_MlirDiagnostic, ?*anyopaque) callconv(.C) c.struct_MlirLogicalResult" = type
      ) do
    {:ok, %Type{zig_t: type, module_name: :MlirDiagnosticHandler}}
  end

  def type_gen("?fn(c.struct_MlirOperation, bool, ?*anyopaque) callconv(.C) void" = type) do
    {:ok, %Type{zig_t: type, module_name: :SymbolTableWalkSymbolTablesCallback}}
  end

  def type_gen("?fn(isize, [*c]c.struct_MlirType, ?*anyopaque) callconv(.C) void" = type) do
    {:ok, %Type{zig_t: type, module_name: :MlirTypesCallback}}
  end

  def type_gen(type) do
    Type.default(type)
  end
end
