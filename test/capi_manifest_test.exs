defmodule Beaver.MLIR.CAPI.ManifestTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR.CAPI
  alias Kinda.CodeGen.{DeclarationManifest, DeclarationSurfaces}

  @moduletag :smoke

  setup_all do
    manifest = CAPI.CodeGen.declaration_manifest()
    declarations = DeclarationManifest.nif_decls(manifest)
    signatures = manifest |> DeclarationManifest.signature_manifest() |> Map.fetch!("entries")

    %{
      declarations: declarations,
      by_name: Map.new(declarations, &{&1.wrapper_name, &1}),
      signatures: Map.new(signatures, &{get_in(&1, ["function", "name"]), &1}),
      emitted_names: MapSet.new(declarations, &Atom.to_string(&1.wrapper_name))
    }
  end

  test "Elixir CAPI recompilation tracks the generated ABI manifest only" do
    expected = [CAPI.CodeGen.declaration_manifest_path()]

    assert CAPI.__info__(:attributes)[:external_resource] == expected
    assert CAPI.Raw.__info__(:attributes)[:external_resource] == expected
  end

  test "declaration manifest matches the generated public and raw surfaces", %{
    declarations: declarations
  } do
    expected = MapSet.new(declarations, &{&1.wrapper_name, arity(&1)})

    raw = CAPI.Raw.__info__(:functions) |> MapSet.new()
    public = CAPI.__info__(:functions) |> MapSet.new()

    assert MapSet.size(expected) > 2_000
    assert MapSet.subset?(expected, raw)
    assert MapSet.subset?(expected, public)

    resolved_declarations =
      CAPI.__kinda_declaration_surfaces__()
      |> DeclarationSurfaces.nif_decls()

    generated = Enum.find(resolved_declarations, &(&1.wrapper_name == :mlirContextCreate))
    assert generated.nif_name != generated.wrapper_name
    refute MapSet.member?(raw, {generated.nif_name, arity(generated)})
    refute MapSet.member?(public, {generated.nif_name, arity(generated)})
  end

  test "declaration manifest preserves signature and wrapper policy metadata", data do
    %{declarations: declarations, by_name: by_name, signatures: signatures} = data
    assert Enum.any?(declarations, &(is_binary(&1.doc) and &1.doc != ""))

    assert Enum.any?(
             declarations,
             &Enum.any?(&1.param_ctypes || [], fn ctype -> ctype != nil end)
           )

    assert Enum.any?(declarations, &(&1.return_ctype != nil))

    for dirty <- [:dirty_cpu, :dirty_io] do
      assert %{dirty: ^dirty} = Map.fetch!(by_name, :"mlirExecutionEngineInvokePacked_#{dirty}")
    end

    for name <- [
          :mlirOperationVerifyWithDiagnostics,
          :mlirTransformApplyNamedSequenceWithDiagnostics
        ] do
      assert %{params: [:context | _], dirty: :dirty_cpu} = Map.fetch!(by_name, name)
    end

    assert function_exported?(CAPI.Raw, :beaver_raw_operation_inventory, 1)
    assert function_exported?(CAPI.Raw, :beaver_raw_peak_rss, 0)
    assert function_exported?(CAPI.Raw, :beaver_raw_process_cpu_time, 0)
    assert function_exported?(CAPI.Raw, :beaver_raw_type_converter_add_conversion_map, 4)

    signature_entry =
      Map.fetch!(signatures, "mlirLLVMDICompileUnitAttrGetWithSourceLanguageDialect")

    assert Enum.at(get_in(signature_entry, ["function", "param_ctypes"]), 5) == %{
             "kind" => "integer",
             "spelling" => "unsigned int"
           }
  end

  test "LogicalResult helpers keep generated surfaces behind the C ABI firewall", data do
    for suffix <- ~w(Failure IsFailure IsSuccess Success) do
      entry = Map.fetch!(data.signatures, "mlirLogicalResult" <> suffix)

      assert entry["generation_blocker_reason"] == nil
      assert entry["variants"] != []

      for prefix <- ["mlir", "beaver"] do
        assert MapSet.member?(data.emitted_names, prefix <> "LogicalResult" <> suffix)
      end
    end
  end

  test "native Zig sources do not call MLIR's inline LogicalResult helpers" do
    forbidden = ~r/c\.mlirLogicalResult(?:Success|Failure|IsSuccess|IsFailure)/

    offenders =
      for path <- Path.wildcard(Path.join(__DIR__, "../native/src/**/*.zig")),
          {line, number} <- path |> File.stream!() |> Stream.with_index(1),
          Regex.match?(forbidden, line),
          do: {path, number}

    assert offenders == []
  end

  test "callback-heavy declarations remain in the callback bridge manifest", data do
    priv_dir = :beaver |> :code.priv_dir() |> List.to_string()

    callback_manifest =
      priv_dir
      |> Path.join("capi_callback_bridge.json")
      |> File.read!()
      |> JSON.decode!()

    assert callback_manifest["version"] == 2
    callback_entries = Map.fetch!(callback_manifest, "entries")

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
    assert [remark_entry] = Map.fetch!(entries_by_runtime, "native_notification")

    assert get_in(remark_entry, ["callback_bridge", "wrapper_name"]) ==
             "beaver_raw_optimization_remarks_attach"

    assert get_in(remark_entry, ["callback_bridge", "lifetime"]) == "remark_engine"
    assert get_in(remark_entry, ["callback_bridge", "owner"]) == "context"

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

    assert pending_entries == []

    # Each collector has one ABI entry, with an explicit scheduler and lifetime.
    collectors = [
      {"beaverCompileLLVMIRToPTX", "compile_llvm_ir_to_ptx", :llvm},
      {"beaverTranslateModuleToLLVMIRText", "translate_module_to_llvm_ir", :llvm},
      {"beaverCompilerKernelLoadAndPopulate", "compiler_kernel_load_and_populate", :kernel},
      {"mlirInferTypeOpInterfaceInferReturnTypes", "infer_return_types", :handles},
      {"mlirInferShapedTypeOpInterfaceInferReturnTypes", "infer_return_type_components",
       :handles},
      {"mlirTransformStateForEachParam", "transform_state_params", :handles},
      {"mlirTransformStateForEachPayloadOp", "transform_state_payload_ops", :handles},
      {"mlirTransformStateForEachPayloadValue", "transform_state_payload_values", :handles},
      {"mlirMemoryEffectsOpInterfaceGetEffects", "memory_effects_query_async", :effects}
    ]

    policies = %{
      llvm: {"dirty_cpu", "native_owner", "nif_call"},
      kernel: {"dirty_io", "stack", "nif_call"},
      handles: {"normal", "stack", "nif_call"},
      effects: {"context_worker", "stack", "async_operation"}
    }

    by_name =
      Map.new(collector_entries, &{get_in(&1, ["function", "name"]), &1["callback_bridge"]})

    assert Enum.sort(Map.keys(by_name)) == Enum.sort(Enum.map(collectors, &elem(&1, 0)))
    assert map_size(by_name) == length(collector_entries)

    for {name, wrapper, policy} <- collectors do
      bridge = Map.fetch!(by_name, name)
      {scheduler, destructor, lifetime} = Map.fetch!(policies, policy)
      assert bridge["wrapper_name"] == "beaver_raw_" <> wrapper
      assert bridge["runtime_backed"]
      assert bridge["scheduler"] == scheduler
      assert bridge["owner"] == "caller"
      assert bridge["destructor"] == destructor
      assert bridge["lifetime"] == lifetime

      if policy == :handles,
        do: assert(bridge["facets"] == ["native_collector", "context_owned_handles"])
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

    assert function_exported?(CAPI.Raw, :beaver_raw_type_converter_add_conversion_map, 4)

    for suffix <-
          ~w(type_converter_add_conversion conversion_pattern_add conversion_target_add_dynamic_op
                     conditionally_speculatable_attach_fallback_model memory_effects_attach_fallback_model
                     pattern_descriptor_op_interface_attach_fallback_model transform_op_interface_attach_fallback_model) do
      assert MapSet.member?(data.emitted_names, "beaver_raw_" <> suffix)
    end

    for name <- Enum.map(runtime_entries, &get_in(&1, ["function", "name"])) do
      entry = Map.fetch!(data.signatures, name)
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

  defp arity(%{params: params}) when is_list(params), do: length(params)
  defp arity(%{params: arity}), do: arity
end
