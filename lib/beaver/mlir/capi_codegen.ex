defmodule Beaver.MLIR.CAPI.CodeGen do
  @moduledoc false
  alias Kinda.CodeGen.{KindDecl, NIFDecl}
  use Kinda.CodeGen

  @impl true
  def type_gen(
        _root_module,
        {:optional_type,
         {:ref,
          [
            :*,
            :const,
            {:fn,
             %Zig.Parser.FnOptions{
               position: nil,
               doc_comment: nil,
               block: nil,
               align: nil,
               linksection: nil,
               callconv: {:enum_literal, :C},
               extern: false,
               export: false,
               pub: false,
               inline: :maybe
             },
             [
               params: [
                 {:_,
                  %Zig.Parser.ParamDeclOption{doc_comment: nil, noalias: false, comptime: false},
                  {:optional_type, {:ref, [:*, :anyopaque]}}}
               ],
               type: :void
             ]}
          ]}} = type
      ) do
    {:ok, %KindDecl{zig_t: type, module_name: Beaver.MLIR.DiagnosticHandlerDeleteUserData}}
  end

  def type_gen(
        _root_module,
        {:optional_type,
         {:ref,
          [
            :*,
            :const,
            {:fn,
             %Zig.Parser.FnOptions{
               position: nil,
               doc_comment: nil,
               block: nil,
               align: nil,
               linksection: nil,
               callconv: {:enum_literal, :C},
               extern: false,
               export: false,
               pub: false,
               inline: :maybe
             },
             [
               params: [
                 {:_,
                  %Zig.Parser.ParamDeclOption{
                    doc_comment: nil,
                    noalias: false,
                    comptime: false
                  }, {:optional_type, {:ref, [:*, :anyopaque]}}},
                 {:_,
                  %Zig.Parser.ParamDeclOption{
                    doc_comment: nil,
                    noalias: false,
                    comptime: false
                  }, :isize},
                 {:_,
                  %Zig.Parser.ParamDeclOption{
                    doc_comment: nil,
                    noalias: false,
                    comptime: false
                  }, :MlirAffineMap}
               ],
               type: :void
             ]}
          ]}} = type
      ) do
    {:ok,
     %KindDecl{zig_t: type, module_name: Beaver.MLIR.AffineMapCompressUnusedSymbolsPopulateResult}}
  end

  def type_gen(
        _root_module,
        {:optional_type,
         {:ref,
          [
            :*,
            :const,
            {:fn,
             %Zig.Parser.FnOptions{
               position: nil,
               doc_comment: nil,
               block: nil,
               align: nil,
               linksection: nil,
               callconv: {:enum_literal, :C},
               extern: false,
               export: false,
               pub: false,
               inline: :maybe
             },
             [
               params: [
                 {:_,
                  %Zig.Parser.ParamDeclOption{
                    doc_comment: nil,
                    noalias: false,
                    comptime: false
                  }, {:optional_type, {:ref, [:*, :anyopaque]}}}
               ],
               type: {:optional_type, {:ref, [:*, :anyopaque]}}
             ]}
          ]}} = type
      ) do
    {:ok, %KindDecl{zig_t: type, module_name: Beaver.MLIR.ExternalPassConstruct}}
  end

  def type_gen(
        _root_module,
        {:optional_type,
         {:ref,
          [
            :*,
            :const,
            {:fn,
             %Zig.Parser.FnOptions{
               position: nil,
               doc_comment: nil,
               block: nil,
               align: nil,
               linksection: nil,
               callconv: {:enum_literal, :C},
               extern: false,
               export: false,
               pub: false,
               inline: :maybe
             },
             [
               params: [
                 {:_,
                  %Zig.Parser.ParamDeclOption{doc_comment: nil, noalias: false, comptime: false},
                  :MlirOperation},
                 {:_,
                  %Zig.Parser.ParamDeclOption{doc_comment: nil, noalias: false, comptime: false},
                  :MlirExternalPass},
                 {:_,
                  %Zig.Parser.ParamDeclOption{doc_comment: nil, noalias: false, comptime: false},
                  {:optional_type, {:ref, [:*, :anyopaque]}}}
               ],
               type: :void
             ]}
          ]}} = type
      ) do
    {:ok, %KindDecl{zig_t: type, module_name: Beaver.MLIR.ExternalPassRun}}
  end

  def type_gen(
        _root_module,
        {:optional_type,
         {:ref,
          [
            :*,
            :const,
            {:fn,
             %Zig.Parser.FnOptions{
               position: nil,
               doc_comment: nil,
               block: nil,
               align: nil,
               linksection: nil,
               callconv: {:enum_literal, :C},
               extern: false,
               export: false,
               pub: false,
               inline: :maybe
             },
             [
               params: [
                 {:_,
                  %Zig.Parser.ParamDeclOption{doc_comment: nil, noalias: false, comptime: false},
                  :MlirOperation},
                 {:_,
                  %Zig.Parser.ParamDeclOption{doc_comment: nil, noalias: false, comptime: false},
                  :bool},
                 {:_,
                  %Zig.Parser.ParamDeclOption{doc_comment: nil, noalias: false, comptime: false},
                  {:optional_type, {:ref, [:*, :anyopaque]}}}
               ],
               type: :void
             ]}
          ]}} = type
      ) do
    {:ok, %KindDecl{zig_t: type, module_name: Beaver.MLIR.SymbolTableWalkSymbolTablesCallback}}
  end

  def type_gen(
        _root_module,
        {:optional_type,
         {:ref,
          [
            :*,
            :const,
            {:fn,
             %Zig.Parser.FnOptions{
               position: nil,
               doc_comment: nil,
               block: nil,
               align: nil,
               linksection: nil,
               callconv: {:enum_literal, :C},
               extern: false,
               export: false,
               pub: false,
               inline: :maybe
             },
             [
               params: [
                 {:_,
                  %Zig.Parser.ParamDeclOption{doc_comment: nil, noalias: false, comptime: false},
                  :MlirContext},
                 {:_,
                  %Zig.Parser.ParamDeclOption{doc_comment: nil, noalias: false, comptime: false},
                  {:optional_type, {:ref, [:*, :anyopaque]}}}
               ],
               type: :MlirLogicalResult
             ]}
          ]}} = type
      ) do
    {:ok, %KindDecl{zig_t: type, module_name: Beaver.MLIR.GenericCallback}}
  end

  def type_gen(_root_module, type)
      when type in [
             :MlirPass,
             :MlirValue,
             :MlirOperation,
             :MlirModule,
             :MlirRegion,
             :MlirAttribute,
             :MlirType,
             :MlirContext,
             :MlirStringRef,
             :MlirLocation,
             :MlirBlock,
             :MlirLogicalResult,
             :MlirAffineMap,
             :MlirNamedAttribute,
             :MlirIdentifier
           ] do
    "Mlir" <> module_name = Atom.to_string(type)
    module_name = Module.concat(Beaver.MLIR, module_name)
    {:ok, %KindDecl{zig_t: type, module_name: module_name}}
  end

  def type_gen(root_module, type) do
    KindDecl.default(root_module, type) |> rewrite_module_name
  end

  defp rewrite_module_name({:ok, %{module_name: module_name} = type}) do
    new_name =
      if module_name in [
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
         ] do
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
      allocated: 1,
      offset: 1
    ]
  end

  @impl true
  def nif_gen({:fn, _fn_opts, [name: :mlirPassManagerRun, params: _params, type: _ret]} = f) do
    %{NIFDecl.from_function(f) | dirty: :cpu}
  end

  def nif_gen(f) do
    NIFDecl.from_function(f)
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
        %KindDecl{
          module_name: Module.concat([Beaver.Native, t, MemRef, rank]),
          kind_functions: memref_kind_functions()
        }
      end

    [
      %KindDecl{
        module_name: Beaver.Native.PtrOwner
      },
      %KindDecl{
        module_name: Beaver.Native.Complex.F32,
        kind_functions: memref_kind_functions()
      }
    ] ++ mem_ref_descriptor_kinds
  end
end
