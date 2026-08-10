defmodule Beaver.MLIR.CAPI.ManifestTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR.CAPI
  alias Kinda.CodeGen.{DeclarationManifest, DeclarationSurfaces}

  @moduletag :smoke

  test "Elixir CAPI recompilation tracks the generated ABI manifest only" do
    expected = [CAPI.CodeGen.declaration_manifest_path()]

    assert CAPI.__info__(:attributes)[:external_resource] == expected
    assert CAPI.Raw.__info__(:attributes)[:external_resource] == expected
  end

  test "declaration manifest matches the generated public and raw surfaces" do
    declaration_manifest = CAPI.CodeGen.declaration_manifest()
    declarations = DeclarationManifest.nif_decls(declaration_manifest)

    expected =
      declarations
      |> Enum.map(fn declaration ->
        arity =
          if is_list(declaration.params), do: length(declaration.params), else: declaration.params

        {declaration.wrapper_name, arity}
      end)
      |> MapSet.new()

    raw = CAPI.Raw.__info__(:functions) |> MapSet.new()
    public = CAPI.__info__(:functions) |> MapSet.new()

    assert MapSet.size(expected) > 2_000
    assert MapSet.subset?(expected, raw)
    assert MapSet.subset?(expected, public)

    resolved_declarations =
      CAPI.__kinda_declaration_surfaces__()
      |> DeclarationSurfaces.nif_decls()

    generated = Enum.find(resolved_declarations, &(&1.wrapper_name == :mlirContextCreate))
    arity = if is_list(generated.params), do: length(generated.params), else: generated.params

    assert generated.nif_name != generated.wrapper_name
    refute MapSet.member?(raw, {generated.nif_name, arity})
    refute MapSet.member?(public, {generated.nif_name, arity})
  end

  test "declaration manifest preserves signature and wrapper policy metadata" do
    declaration_manifest = CAPI.CodeGen.declaration_manifest()
    declarations = DeclarationManifest.nif_decls(declaration_manifest)

    assert Enum.any?(declarations, &(is_binary(&1.doc) and &1.doc != ""))

    assert Enum.any?(
             declarations,
             &Enum.any?(&1.param_ctypes || [], fn ctype -> ctype != nil end)
           )

    assert Enum.any?(declarations, &(&1.return_ctype != nil))

    assert %{dirty: :dirty_cpu} =
             Enum.find(
               declarations,
               &(&1.wrapper_name == :mlirExecutionEngineInvokePacked_dirty_cpu)
             )

    assert %{dirty: :dirty_io} =
             Enum.find(
               declarations,
               &(&1.wrapper_name == :mlirExecutionEngineInvokePacked_dirty_io)
             )

    assert %{params: [:context | _], dirty: :dirty_cpu} =
             Enum.find(declarations, &(&1.wrapper_name == :mlirOperationVerifyWithDiagnostics))

    assert %{params: [:context | _], dirty: :dirty_cpu} =
             Enum.find(
               declarations,
               &(&1.wrapper_name == :mlirTransformApplyNamedSequenceWithDiagnostics)
             )

    signature_entry =
      declaration_manifest
      |> DeclarationManifest.signature_manifest()
      |> Map.fetch!("entries")
      |> Enum.find(
        &(get_in(&1, ["function", "name"]) ==
            "mlirLLVMDICompileUnitAttrGetWithSourceLanguageDialect")
      )

    if signature_entry do
      assert Enum.at(get_in(signature_entry, ["function", "param_ctypes"]), 5) == %{
               "kind" => "integer",
               "spelling" => "unsigned int"
             }
    else
      refute function_exported?(
               CAPI,
               :mlirLLVMDICompileUnitAttrGetWithSourceLanguageDialect,
               15
             )
    end
  end

  test "LogicalResult helpers keep generated surfaces behind the C ABI firewall" do
    declaration_manifest = CAPI.CodeGen.declaration_manifest()

    inline_helpers = [
      "mlirLogicalResultFailure",
      "mlirLogicalResultIsFailure",
      "mlirLogicalResultIsSuccess",
      "mlirLogicalResultSuccess"
    ]

    signature_entries =
      declaration_manifest
      |> DeclarationManifest.signature_manifest()
      |> Map.fetch!("entries")

    for name <- inline_helpers do
      entry = Enum.find(signature_entries, &(get_in(&1, ["function", "name"]) == name))

      assert entry["generation_blocker_reason"] == nil
      assert entry["variants"] != []
    end

    generated_names =
      declaration_manifest
      |> DeclarationManifest.nif_decls()
      |> Enum.map(&Atom.to_string(&1.wrapper_name))
      |> MapSet.new()

    for name <-
          inline_helpers ++
            [
              "beaverLogicalResultFailure",
              "beaverLogicalResultIsFailure",
              "beaverLogicalResultIsSuccess",
              "beaverLogicalResultSuccess"
            ] do
      assert MapSet.member?(generated_names, name)
    end
  end

  test "native Zig sources do not call MLIR's inline LogicalResult helpers" do
    forbidden = ~r/c\.mlirLogicalResult(?:Success|Failure|IsSuccess|IsFailure)/

    offenders =
      __DIR__
      |> Path.join("../native/src/**/*.zig")
      |> Path.wildcard()
      |> Enum.flat_map(fn path ->
        path
        |> File.stream!()
        |> Stream.with_index(1)
        |> Enum.flat_map(fn {line, line_number} ->
          if Regex.match?(forbidden, line), do: [{path, line_number}], else: []
        end)
      end)

    assert offenders == []
  end

  test "callback-heavy declarations remain in the callback bridge manifest" do
    priv_dir = :beaver |> :code.priv_dir() |> List.to_string()

    callback_manifest =
      priv_dir
      |> Path.join("capi_callback_bridge.json")
      |> File.read!()
      |> JSON.decode!()

    assert callback_manifest["version"] == 2
    callback_entries = Map.fetch!(callback_manifest, "entries")

    declaration_manifest = CAPI.CodeGen.declaration_manifest()

    emitted_names =
      declaration_manifest
      |> DeclarationManifest.nif_decls()
      |> Enum.map(&Atom.to_string(&1.wrapper_name))
      |> MapSet.new()

    assert callback_entries != []

    assert Enum.any?(
             callback_entries,
             &(get_in(&1, ["function", "name"]) == "mlirValueReplaceUsesWithIf")
           )

    entries_by_runtime =
      Enum.group_by(callback_entries, &get_in(&1, ["callback_bridge", "runtime"]))

    runtime_entries = Map.fetch!(entries_by_runtime, "dispatcher")
    collector_entries = Map.fetch!(entries_by_runtime, "native_collector")
    manual_runtime_entries = Map.fetch!(entries_by_runtime, "manual_async_callback")
    pending_entries = Map.get(entries_by_runtime, "pending", [])

    assert Enum.map(runtime_entries, &get_in(&1, ["function", "name"])) |> MapSet.new() ==
             MapSet.new([
               "mlirConversionTargetAddDynamicallyLegalDialect",
               "mlirConversionTargetAddDynamicallyLegalOp",
               "mlirConversionTargetMarkOpRecursivelyLegal",
               "mlirConversionTargetMarkUnknownOpDynamicallyLegal",
               "mlirOpConversionPatternCreate",
               "mlirTypeConverterAdd1ToNConversion",
               "mlirTypeConverterAdd1ToNTargetMaterialization",
               "mlirTypeConverterAddConversion",
               "mlirTypeConverterAddSourceMaterialization",
               "mlirTypeConverterAddTargetMaterialization",
               "mlirConditionallySpeculatableOpInterfaceAttachFallbackModel",
               "mlirMemoryEffectsOpInterfaceAttachFallbackModel",
               "mlirPatternDescriptorOpInterfaceAttachFallbackModel",
               "mlirTransformOpInterfaceAttachFallbackModel",
               "mlirDynamicOpTraitCreate"
             ])

    for entry <- pending_entries do
      name = get_in(entry, ["function", "name"])
      assert get_in(entry, ["callback_bridge", "reason"]) == "callback_bridge_required"
      refute MapSet.member?(emitted_names, name)
    end

    assert pending_entries == []

    assert Enum.map(collector_entries, &get_in(&1, ["callback_bridge", "wrapper_name"]))
           |> MapSet.new() ==
             MapSet.new([
               "beaver_raw_compile_llvm_ir_to_ptx",
               "beaver_raw_translate_module_to_llvm_ir",
               "beaver_raw_infer_return_types",
               "beaver_raw_infer_return_type_components",
               "beaver_raw_transform_state_params",
               "beaver_raw_transform_state_payload_ops",
               "beaver_raw_transform_state_payload_values"
             ])

    {llvm_entries, transform_collector_entries} =
      Enum.split_with(
        collector_entries,
        &(get_in(&1, ["function", "name"]) in [
            "beaverCompileLLVMIRToPTX",
            "beaverTranslateModuleToLLVMIRText"
          ])
      )

    for entry <- transform_collector_entries do
      bridge = Map.fetch!(entry, "callback_bridge")
      assert bridge["runtime_backed"]
      assert bridge["scheduler"] == "normal"
      assert bridge["owner"] == "caller"
      assert bridge["destructor"] == "stack"
      assert bridge["lifetime"] == "nif_call"
      assert bridge["facets"] == ["native_collector", "context_owned_handles"]
    end

    assert Enum.map(llvm_entries, &get_in(&1, ["callback_bridge", "wrapper_name"]))
           |> MapSet.new() ==
             MapSet.new([
               "beaver_raw_compile_llvm_ir_to_ptx",
               "beaver_raw_translate_module_to_llvm_ir"
             ])

    for entry <- llvm_entries do
      assert %{
               "runtime_backed" => true,
               "scheduler" => "dirty_cpu",
               "owner" => "caller",
               "destructor" => "native_owner",
               "lifetime" => "nif_call"
             } = entry["callback_bridge"]
    end

    assert [manual_runtime] = manual_runtime_entries
    assert get_in(manual_runtime, ["function", "name"]) == "mlirValueReplaceUsesWithIf"

    assert %{
             "wrapper_name" => "beaver_raw_value_replace_uses_with_if",
             "runtime_backed" => true,
             "scheduler" => "context_worker",
             "owner" => "beam_process",
             "destructor" => "native_owner",
             "lifetime" => "async_operation"
           } = manual_runtime["callback_bridge"]

    for entry <- runtime_entries do
      bridge = Map.fetch!(entry, "callback_bridge")
      assert bridge["reason"] == nil
      assert bridge["runtime"] == "dispatcher"
      assert bridge["scheduler"] == "foreign_thread"
      assert bridge["owner"] == "beam_process"
      assert bridge["destructor"] == "native_owner"
      assert bridge["lifetime"] == "native_owner"
      assert bridge["timeout_ms"] == 30_000
    end

    assert MapSet.member?(emitted_names, "beaver_raw_type_converter_add_conversion")
    assert MapSet.member?(emitted_names, "beaver_raw_conversion_pattern_add")
    assert MapSet.member?(emitted_names, "beaver_raw_conversion_target_add_dynamic_op")

    assert MapSet.member?(
             emitted_names,
             "beaver_raw_conditionally_speculatable_attach_fallback_model"
           )

    assert MapSet.member?(emitted_names, "beaver_raw_memory_effects_attach_fallback_model")

    assert MapSet.member?(
             emitted_names,
             "beaver_raw_pattern_descriptor_op_interface_attach_fallback_model"
           )

    assert MapSet.member?(
             emitted_names,
             "beaver_raw_transform_op_interface_attach_fallback_model"
           )

    signature_entries =
      declaration_manifest
      |> DeclarationManifest.signature_manifest()
      |> Map.fetch!("entries")

    for name <- Enum.map(runtime_entries, &get_in(&1, ["function", "name"])) do
      entry = Enum.find(signature_entries, &(get_in(&1, ["function", "name"]) == name))
      assert entry["generation_blocker_reason"] == nil
      assert get_in(entry, ["callback_bridge", "runtime_backed"])
      assert [_resolved_variant] = entry["variants"]
    end
  end

  test "build does not install generated wrapper sources" do
    priv_dir = :beaver |> :code.priv_dir() |> List.to_string()

    refute File.exists?(Path.join(priv_dir, "capi_functions.ex"))
    refute File.exists?(Path.join(priv_dir, "generated/wrapper.zig"))
  end
end
