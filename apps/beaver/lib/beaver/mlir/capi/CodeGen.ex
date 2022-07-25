defmodule Beaver.MLIR.CAPI.CodeGen do
  alias Fizz.CodeGen.{Type, NIF, Function}

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

  def type_gen(root_module, type) do
    Type.default(root_module, type)
  end

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
end
