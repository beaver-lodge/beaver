defmodule Beaver.CAPI.ManifestGenerator do
  @moduledoc false

  @policy_markers [
    {"BeaverCapiPolicyDiagnostics__", "diagnostics"},
    {"BeaverCapiPolicyDirtyCPUAndIO__", "dirty_cpu_io"},
    {"BeaverCapiPolicyCallbackBridge__", "callback_bridge"},
    {"BeaverCapiPolicyCallbackRuntime__", "callback_runtime"},
    {"BeaverCapiPolicyExclude__", "exclude"}
  ]

  @runtime_declarations %{
    "mlirTypeConverterAddConversion" => %{
      name: "beaver_raw_type_converter_create_callback",
      params: ["callback", "timeout_ms"]
    },
    "mlirConditionallySpeculatableOpInterfaceAttachFallbackModel" => %{
      name: "beaver_raw_conditionally_speculatable_attach_fallback_model",
      params: ["context", "operation_name", "callback", "timeout_ms"]
    }
  }

  def run(argv) do
    {opts, _, _} =
      OptionParser.parse(argv,
        strict: [declaration: :string, callback_bridge: :string]
      )

    ast = IO.read(:stdio, :eof) |> decode_json!()
    policy = policy_from_ast(ast)

    functions =
      ast
      |> collect_functions()
      |> Enum.filter(&candidate?/1)
      |> Enum.sort_by(& &1.name)

    declaration_manifest = declaration_manifest(functions, policy)
    callback_bridge_manifest = callback_bridge_manifest(functions, policy)

    opts
    |> Keyword.fetch!(:declaration)
    |> write_json!(declaration_manifest)

    opts
    |> Keyword.fetch!(:callback_bridge)
    |> write_json!(callback_bridge_manifest)
  end

  defp policy_from_ast(ast) do
    ast
    |> collect_enum_constants()
    |> Enum.reduce(%{}, fn marker, policy ->
      case decode_policy_marker(marker) do
        {category, name} ->
          Map.update(policy, category, MapSet.new([name]), &MapSet.put(&1, name))

        nil ->
          policy
      end
    end)
  end

  defp collect_enum_constants(nodes) when is_list(nodes),
    do: Enum.flat_map(nodes, &collect_enum_constants/1)

  defp collect_enum_constants(%{"kind" => "EnumConstantDecl", "name" => name} = node),
    do: [name | collect_enum_constants(Map.get(node, "inner", []))]

  defp collect_enum_constants(%{} = node),
    do: collect_enum_constants(Map.get(node, "inner", []))

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

  defp candidate?(%{name: name}) do
    String.starts_with?(name, "mlir") or String.starts_with?(name, "beaver")
  end

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
        Enum.flat_map(functions, fn function ->
          function
          |> variants(policy)
          |> Enum.map(&nif_decl(function, &1))
        end),
      "type_decls" => []
    }
  end

  defp callback_bridge_manifest(functions, policy) do
    entries =
      for function <- functions,
          callback_bridge?(policy, function.name) do
        runtime_backed = policy_member?(policy, "callback_runtime", function.name)

        %{
          "function" => %{
            "name" => function.name,
            "arity" => function.arity,
            "params" => function.params,
            "doc" => function.doc,
            "param_ctypes" => function.param_ctypes,
            "return_ctype" => function.return_ctype
          },
          "callback_bridge" => %{
            "function" => function.name,
            "reason" => if(runtime_backed, do: nil, else: "callback_bridge_required"),
            "unblock_path" => if(runtime_backed, do: nil, else: "callback_bridge_runtime"),
            "scheduler" => if(runtime_backed, do: "foreign_thread", else: "unspecified"),
            "facets" => [
              "beam_callback",
              "lifetime_contract",
              "scheduler_contract",
              "rich_input_decoder"
            ],
            "runtime" => if(runtime_backed, do: "dispatcher", else: "pending"),
            "runtime_backed" => runtime_backed,
            "owner" => if(runtime_backed, do: "beam_process", else: "unspecified"),
            "destructor" => if(runtime_backed, do: "native_owner", else: "unspecified"),
            "lifetime" => if(runtime_backed, do: "native_owner", else: "unspecified"),
            "timeout_ms" => if(runtime_backed, do: 30_000, else: nil)
          }
        }
      end

    %{"version" => 2, "entries" => entries}
  end

  defp signature_entry(function, policy) do
    entry = %{
      "function" => %{
        "name" => function.name,
        "arity" => function.arity,
        "params" => function.params,
        "doc" => function.doc,
        "param_ctypes" => function.param_ctypes,
        "return_ctype" => function.return_ctype
      },
      "generation_blocker_reason" => blocker_reason(function.name, policy),
      "variants" => Enum.map(variants(function, policy), &signature_variant(function, &1))
    }

    if callback_bridge?(policy, function.name) do
      callback_entry =
        callback_bridge_manifest([function], policy)
        |> Map.fetch!("entries")
        |> List.first()
        |> Map.fetch!("callback_bridge")

      Map.put(entry, "callback_bridge", callback_entry)
    else
      entry
    end
  end

  defp blocker_reason(name, policy) do
    cond do
      policy_member?(policy, "callback_bridge", name) -> "callback_bridge_required"
      policy_member?(policy, "exclude", name) -> "consumer_policy"
      true -> nil
    end
  end

  defp variants(function, policy) do
    name = function.name

    cond do
      policy_member?(policy, "callback_runtime", name) ->
        [
          Map.fetch!(@runtime_declarations, name)
          |> Map.put(:dirty, false)
          |> Map.put(:runtime, true)
        ]

      blocker_reason(name, policy) != nil ->
        []

      diagnostics?(name, policy) ->
        [
          %{name: name, params: function.params, dirty: false},
          %{name: name <> "WithDiagnostics", params: ["context" | function.params], dirty: false}
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

  defp diagnostics?(name, policy) do
    String.ends_with?(name, "GetChecked") or policy_member?(policy, "diagnostics", name)
  end

  defp policy_member?(policy, category, name) do
    policy
    |> Map.get(category, MapSet.new())
    |> MapSet.member?(name)
  end

  defp callback_bridge?(policy, name) do
    policy_member?(policy, "callback_bridge", name) or
      policy_member?(policy, "callback_runtime", name)
  end

  defp nif_decl(function, variant) do
    %{
      "wrapper_name" => variant.name,
      "nif_name" => nil,
      "params" => variant.params,
      "doc" => function.doc,
      "param_ctypes" => if(variant[:runtime], do: [], else: function.param_ctypes),
      "return_ctype" => if(variant[:runtime], do: nil, else: function.return_ctype),
      "param_typespecs" => nil,
      "return_typespec" => nil,
      "dirty" => variant.dirty
    }
  end

  defp signature_variant(_function, variant) do
    %{
      "wrapper_name" => variant.name,
      "params" => variant.params,
      "doc" => nil,
      "dirty" => variant.dirty,
      "param_typespecs" => nil,
      "return_typespec" => nil
    }
  end

  defp collect_functions(nodes) when is_list(nodes),
    do: Enum.flat_map(nodes, &collect_functions/1)

  defp collect_functions(%{"kind" => "FunctionDecl", "name" => name} = node) do
    params =
      node
      |> Map.get("inner", [])
      |> Enum.with_index()
      |> Enum.filter(fn {child, _index} -> Map.get(child, "kind") == "ParmVarDecl" end)
      |> Enum.map(fn {child, index} ->
        %{
          name: Map.get(child, "name", "param_#{index}"),
          ctype: ctype(Map.get(child, "type"))
        }
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

  defp ctype(%{} = type) do
    type
    |> then(&(Map.get(&1, "desugaredQualType") || Map.get(&1, "qualType")))
    |> ctype()
  end

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
    type in [
      "char",
      "signed char",
      "unsigned char",
      "short",
      "unsigned short",
      "int",
      "unsigned int",
      "long",
      "unsigned long",
      "long long",
      "unsigned long long",
      "i8",
      "i16",
      "i32",
      "i64",
      "u8",
      "u16",
      "u32",
      "u64",
      "isize",
      "usize",
      "c_int",
      "c_uint",
      "size_t",
      "ssize_t",
      "intptr_t",
      "uintptr_t",
      "ptrdiff_t"
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
       when kind in ["TextComment", "VerbatimBlockLineComment", "VerbatimLineComment"],
       do: normalize_doc(text)

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

  defp decode_json!(json) do
    cond do
      Code.ensure_loaded?(JSON) and function_exported?(JSON, :decode!, 1) ->
        apply(JSON, :decode!, [json])

      Code.ensure_loaded?(Jason) and function_exported?(Jason, :decode!, 1) ->
        apply(Jason, :decode!, [json])

      true ->
        raise "JSON decoder unavailable; pass the Jason ebin directory with elixir -pa"
    end
  end

  defp write_json!(path, data) do
    encoded =
      cond do
        Code.ensure_loaded?(JSON) and function_exported?(JSON, :encode!, 1) ->
          apply(JSON, :encode!, [data])

        Code.ensure_loaded?(Jason) and function_exported?(Jason, :encode!, 2) ->
          apply(Jason, :encode!, [data, [pretty: true]])

        true ->
          raise "JSON encoder unavailable; pass the Jason ebin directory with elixir -pa"
      end

    path = Path.expand(path)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, encoded)
  end
end

Beaver.CAPI.ManifestGenerator.run(System.argv())
