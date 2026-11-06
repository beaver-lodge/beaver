defmodule Beaver.CAPI.ManifestGenerator do
  @moduledoc false

  @policy_markers [
    {"BeaverCapiPolicyDiagnostics__", "diagnostics"},
    {"BeaverCapiPolicyDirtyCPUAndIO__", "dirty_cpu_io"},
    {"BeaverCapiPolicyCallbackBridge__", "callback_bridge"},
    {"BeaverCapiPolicyCallbackRuntime__", "callback_runtime"},
    {"BeaverCapiPolicyManualAdapter__", "manual_adapter"},
    {"BeaverCapiPolicyManualRuntime__", "manual_runtime"},
    {"BeaverCapiPolicyExclude__", "exclude"}
  ]

  @bridge_defaults %{
    "runtime" => "native_collector",
    "scheduler" => "normal",
    "owner" => "caller",
    "lifetime" => "nif_call",
    "destructor" => "stack"
  }
  @manual_bridges Map.new(
                    [
                      {"beaverTranslateModuleToLLVMIRText", "translate_module_to_llvm_ir",
                       %{"scheduler" => "dirty_cpu", "destructor" => "native_owner"}},
                      {"beaverCompileLLVMIRToPTX", "compile_llvm_ir_to_ptx",
                       %{"scheduler" => "dirty_cpu", "destructor" => "native_owner"}},
                      {"beaverCompilerKernelLoadAndPopulate", "compiler_kernel_load_and_populate",
                       %{"scheduler" => "dirty_io"}},
                      {"beaverMemoryEffectsOpInterfaceGetEffects", "memory_effects_query_async",
                       %{"scheduler" => "context_worker", "lifetime" => "async_operation"}},
                      {"mlirValueReplaceUsesWithIf", "value_replace_uses_with_if",
                       %{
                         "runtime" => "manual_async_callback",
                         "scheduler" => "context_worker",
                         "owner" => "beam_process",
                         "lifetime" => "async_operation",
                         "destructor" => "native_owner"
                       }}
                    ],
                    fn {source, wrapper, overrides} ->
                      policy =
                        @bridge_defaults |> Map.merge(overrides) |> Map.put("wrapper_name", "beaver_raw_" <> wrapper)

                      {source, policy}
                    end
                  )

  @runtime_declarations Map.new(
                          [
                            {"mlirTypeConverterAddConversion", "type_converter_add_conversion",
                             ~w(registration callback timeout_ms)},
                            {"mlirTypeConverterAdd1ToNConversion", "type_converter_add_1_to_n_conversion",
                             ~w(registration callback timeout_ms)},
                            {"mlirTypeConverterAddSourceMaterialization", "type_converter_add_source_materialization",
                             ~w(registration callback timeout_ms)},
                            {"mlirTypeConverterAddTargetMaterialization", "type_converter_add_target_materialization",
                             ~w(registration callback timeout_ms)},
                            {"mlirTypeConverterAdd1ToNTargetMaterialization",
                             "type_converter_add_1_to_n_target_materialization", ~w(registration callback timeout_ms)},
                            {"mlirConversionTargetAddDynamicallyLegalOp", "conversion_target_add_dynamic_op",
                             ~w(registration operation_name callback timeout_ms)},
                            {"mlirConversionTargetAddDynamicallyLegalDialect", "conversion_target_add_dynamic_dialect",
                             ~w(registration dialect_name callback timeout_ms)},
                            {"mlirConversionTargetMarkOpRecursivelyLegal", "conversion_target_mark_recursively_legal",
                             ~w(registration operation_name callback timeout_ms)},
                            {"mlirConversionTargetMarkUnknownOpDynamicallyLegal",
                             "conversion_target_mark_unknown_dynamic", ~w(registration callback timeout_ms)},
                            {"mlirOpConversionPatternCreate", "conversion_pattern_add",
                             ~w(pattern_set root_name benefit context type_converter_registration callback one_to_n timeout_ms)},
                            {"mlirConditionallySpeculatableOpInterfaceAttachFallbackModel",
                             "conditionally_speculatable_attach_fallback_model",
                             ~w(context operation_name callback timeout_ms)},
                            {"beaverMemoryEffectsOpInterfaceAttachFallbackModel",
                             "memory_effects_attach_fallback_model", ~w(context operation_name callback timeout_ms)}
                          ],
                          fn {source, name, params} -> {source, %{name: "beaver_raw_" <> name, params: params}} end
                        )

  def run(argv) do
    {opts, _, _} =
      OptionParser.parse(argv, strict: [declaration: :string, callback_bridge: :string, ast: :string])

    ast =
      case Keyword.get(opts, :ast) do
        nil -> IO.binread(:stdio, :eof)
        path -> File.read!(path)
      end
      |> decode_json!()

    policy = policy_from_ast(ast)

    functions = ast |> collect_functions() |> Enum.filter(&candidate?/1) |> Enum.sort_by(& &1.name)

    declaration_manifest = declaration_manifest(functions, policy)
    callback_bridge_manifest = callback_bridge_manifest(functions, policy)

    opts |> Keyword.fetch!(:declaration) |> write_json!(declaration_manifest)

    opts |> Keyword.fetch!(:callback_bridge) |> write_json!(callback_bridge_manifest)
  end

  defp policy_from_ast(ast) do
    ast
    |> collect_enum_constants()
    |> Enum.reduce(%{}, fn marker, policy ->
      case decode_policy_marker(marker) do
        {category, name} -> Map.update(policy, category, MapSet.new([name]), &MapSet.put(&1, name))
        nil -> policy
      end
    end)
  end

  defp collect_enum_constants(nodes) when is_list(nodes), do: Enum.flat_map(nodes, &collect_enum_constants/1)

  defp collect_enum_constants(%{"kind" => "EnumConstantDecl", "name" => name} = node),
    do: [name | collect_enum_constants(Map.get(node, "inner", []))]

  defp collect_enum_constants(%{} = node), do: collect_enum_constants(Map.get(node, "inner", []))

  defp collect_enum_constants(_node), do: []

  defp decode_policy_marker(marker) do
    Enum.find_value(@policy_markers, fn {prefix, category} ->
      if String.starts_with?(marker, prefix) do
        case String.replace_prefix(marker, prefix, "") do
          "" -> nil
          name -> {category, name}
        end
      end
    end)
  end

  defp candidate?(%{name: name}), do: String.starts_with?(name, "mlir") or String.starts_with?(name, "beaver")

  defp declaration_manifest(functions, policy) do
    signature_manifest = %{
      "version" => 1,
      "records" => [],
      "entries" => Enum.map(functions, &signature_entry(&1, policy))
    }

    %{
      "version" => 1,
      "signature_manifest_version" => 1,
      "signature_manifest" => signature_manifest,
      "nif_decls" =>
        Enum.flat_map(functions, fn function -> function |> variants(policy) |> Enum.map(&nif_decl(function, &1)) end),
      "type_decls" => []
    }
  end

  defp callback_bridge_manifest(functions, policy) do
    entries =
      for function <- functions,
          callback_bridge?(policy, function.name),
          do: %{"function" => function_metadata(function), "callback_bridge" => bridge_entry(function.name, policy)}

    %{"version" => 2, "entries" => entries}
  end

  defp bridge_entry(name, policy) do
    bridge = bridge_metadata(name, policy)

    defaults = %{
      "function" => name,
      "reason" => if(bridge, do: nil, else: "callback_bridge_required"),
      "unblock_path" => if(bridge, do: nil, else: "callback_bridge_runtime"),
      "runtime" => "pending",
      "runtime_backed" => bridge != nil,
      "wrapper_name" => nil,
      "timeout_ms" => nil,
      "scheduler" => "unspecified",
      "owner" => "unspecified",
      "destructor" => "unspecified",
      "lifetime" => "unspecified"
    }

    defaults |> Map.merge(bridge || %{}) |> Map.merge(bridge_facets(bridge))
  end

  defp function_metadata(function),
    do: Map.new(function, fn {key, value} -> {Atom.to_string(key), value} end)

  defp signature_entry(function, policy) do
    entry = %{
      "function" => function_metadata(function),
      "generation_blocker_reason" => blocker_reason(function.name, policy),
      "variants" => Enum.map(variants(function, policy), &signature_variant(function, &1))
    }

    if callback_bridge?(policy, function.name) do
      Map.put(entry, "callback_bridge", bridge_entry(function.name, policy))
    else
      entry
    end
  end

  defp blocker_reason(name, policy) do
    cond do
      policy_member?(policy, "callback_bridge", name) -> "callback_bridge_required"
      policy_member?(policy, "manual_adapter", name) -> "manual_adapter"
      policy_member?(policy, "manual_runtime", name) -> "manual_runtime"
      policy_member?(policy, "exclude", name) -> "consumer_policy"
      true -> nil
    end
  end

  defp variants(function, policy) do
    name = function.name

    cond do
      policy_member?(policy, "callback_runtime", name) ->
        [Map.fetch!(@runtime_declarations, name) |> Map.put(:dirty, false) |> Map.put(:runtime, true)]

      blocker_reason(name, policy) != nil ->
        []

      diagnostics?(name, policy) ->
        [
          %{name: name, params: function.params, dirty: false},
          %{name: name <> "WithDiagnostics", params: ["context" | function.params], dirty: "dirty_cpu"}
        ]

      policy_member?(policy, "dirty_cpu_io", name) ->
        [
          %{name: name, params: function.params, dirty: false},
          %{name: name <> "_dirty_io", params: function.params, dirty: "dirty_io"},
          %{name: name <> "_dirty_cpu", params: function.params, dirty: "dirty_cpu"}
        ]

      true ->
        [%{name: name, params: function.params, dirty: false}]
    end
  end

  defp diagnostics?(name, policy),
    do: String.ends_with?(name, "GetChecked") or policy_member?(policy, "diagnostics", name)

  defp policy_member?(policy, category, name), do: policy |> Map.get(category, MapSet.new()) |> MapSet.member?(name)

  defp callback_bridge?(policy, name) do
    policy_member?(policy, "callback_bridge", name) or policy_member?(policy, "callback_runtime", name) or
      policy_member?(policy, "manual_adapter", name) or policy_member?(policy, "manual_runtime", name)
  end

  defp bridge_metadata(name, policy) do
    cond do
      policy_member?(policy, "callback_runtime", name) ->
        %{
          "wrapper_name" => Map.fetch!(@runtime_declarations, name).name,
          "runtime" => "dispatcher",
          "scheduler" => "foreign_thread",
          "owner" => "beam_process",
          "destructor" => "native_owner",
          "lifetime" => "native_owner",
          "timeout_ms" => 30_000
        }

      policy_member?(policy, "manual_adapter", name) or
          policy_member?(policy, "manual_runtime", name) ->
        @manual_bridges |> Map.fetch!(name) |> Map.put("timeout_ms", nil)

      true ->
        nil
    end
  end

  defp bridge_facets(%{"runtime" => "native_collector"}),
    do: %{"facets" => ["native_collector", "context_owned_handles"]}

  defp bridge_facets(_bridge),
    do: %{"facets" => ~w(beam_callback lifetime_contract scheduler_contract rich_input_decoder)}

  defp nif_decl(function, variant) do
    Map.merge(signature_variant(function, variant), %{
      "nif_name" => nil,
      "doc" => function.doc,
      "param_ctypes" => if(variant[:runtime], do: [], else: function.param_ctypes),
      "return_ctype" => if(variant[:runtime], do: nil, else: function.return_ctype)
    })
  end

  defp signature_variant(_function, variant),
    do: %{
      "wrapper_name" => variant.name,
      "params" => variant.params,
      "doc" => nil,
      "dirty" => variant.dirty,
      "param_typespecs" => nil,
      "return_typespec" => nil
    }

  defp collect_functions(nodes) when is_list(nodes), do: Enum.flat_map(nodes, &collect_functions/1)

  defp collect_functions(%{"kind" => "FunctionDecl", "name" => name} = node) do
    params =
      node
      |> Map.get("inner", [])
      |> Enum.with_index()
      |> Enum.filter(fn {child, _index} -> Map.get(child, "kind") == "ParmVarDecl" end)
      |> Enum.map(fn {child, index} ->
        %{name: Map.get(child, "name", "param_#{index}"), ctype: ctype(Map.get(child, "type"))}
      end)

    function = %{
      name: name,
      params: Enum.map(params, & &1.name),
      param_ctypes: Enum.map(params, & &1.ctype),
      arity: length(params),
      doc: extract_doc(node),
      return_ctype: return_ctype(node)
    }

    [function | collect_functions(Map.get(node, "inner", []))]
  end

  defp collect_functions(%{} = node), do: collect_functions(Map.get(node, "inner", []))
  defp collect_functions(_node), do: []

  defp ctype(nil), do: nil

  defp ctype(%{} = type), do: type |> then(&(Map.get(&1, "desugaredQualType") || Map.get(&1, "qualType"))) |> ctype()

  defp ctype(type) when is_binary(type) do
    spelling = normalize_type(type)
    %{"spelling" => spelling, "kind" => classify_type(spelling)}
  end

  defp return_ctype(node) do
    node
    |> Map.get("type", %{})
    |> then(&(Map.get(&1, "desugaredQualType") || Map.get(&1, "qualType")))
    |> case do
      nil -> nil
      type -> type |> normalize_type() |> String.split(" (", parts: 2) |> hd() |> ctype()
    end
  end

  defp normalize_type(type) do
    type
    |> String.replace(~r/\b(const|volatile|restrict)\b/u, "")
    |> String.replace(~r/\s+/, " ")
    |> String.replace(" *", "*")
    |> String.replace("* ", "*")
    |> String.trim()
  end

  defp classify_type("void"), do: "void"
  defp classify_type("bool"), do: "bool"
  defp classify_type(type) when type in ["float", "double", "f32", "f64"], do: "float"

  defp classify_type(type) do
    cond do
      String.contains?(type, "(*)") or String.contains?(type, "(*") -> "function_pointer"
      String.contains?(type, "*") -> "pointer"
      String.starts_with?(type, "enum ") -> "enum"
      String.starts_with?(type, "struct ") or String.starts_with?(type, "union ") -> "record"
      integer_type?(type) -> "integer"
      true -> "unknown"
    end
  end

  defp integer_type?(type) do
    type in ~w(
      char short int long i8 i16 i32 i64 u8 u16 u32 u64 isize usize c_int c_uint size_t ssize_t intptr_t
      uintptr_t ptrdiff_t
    ) or
      type in [
        "signed char",
        "unsigned char",
        "unsigned short",
        "unsigned int",
        "unsigned long",
        "long long",
        "unsigned long long"
      ] or Regex.match?(~r/^u?int(8|16|32|64)_t$/u, type)
  end

  defp extract_doc(node) do
    node
    |> Map.get("inner", [])
    |> Enum.find(&(Map.get(&1, "kind") == "FullComment"))
    |> render_comment()
    |> case do
      "" -> nil
      doc -> doc
    end
  end

  defp render_comment(nil), do: ""

  defp render_comment(%{"kind" => kind, "text" => text})
       when kind in ["TextComment", "VerbatimBlockLineComment", "VerbatimLineComment"], do: normalize_doc(text)

  defp render_comment(%{"kind" => "ParamCommandComment", "param" => param} = node),
    do: normalize_doc("@param #{param} #{render_children(node)}")

  defp render_comment(%{"kind" => "BlockCommandComment", "name" => name} = node),
    do: normalize_doc("@#{name} #{render_children(node)}")

  defp render_comment(%{} = node), do: render_children(node)
  defp render_comment(_node), do: ""

  defp render_children(node) do
    node
    |> Map.get("inner", [])
    |> Enum.map(&render_comment/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.join("\n")
    |> normalize_doc()
  end

  defp normalize_doc(text), do: text |> String.replace(~r/[ \t]+/, " ") |> String.trim()

  defp decode_json!(json), do: JSON.decode!(json)

  defp write_json!(path, data) do
    encoded = JSON.encode!(data)
    path = Path.expand(path)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, encoded)
  end
end

Beaver.CAPI.ManifestGenerator.run(System.argv())
