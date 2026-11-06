defmodule Beaver.MLIR.Conversion.Ex.CRuntime do
  @moduledoc """
  Batata BTT ABI 12 signatures and C adapters for raw buffers and owned SSA handles.
  Compile `adapter_source/1` with an admitted library, matching `aot.h` and
  `-fvisibility=hidden`; use one artifact namespace for the adapter and `Ex.plan/1`.
  Aggregates pass by address for native C ABI classification. Statuses, ownership,
  bounded stepping and cleanup stay explicit; see [runtime contracts](dialect-conversion.html#ex-provider-boundary).
  """
  alias Beaver.MLIR.Type
  @abi 0x425454000000000C
  @aggregates ~w(BttEntry BttProcess BttValue BttCopy BttImport)
  @execution ~w(process call step event reply result cancel drain term_start copy_begin copy_step copy_cancel)
  @process ["BttRuntime *", "BttProcess"]
  @value ["BttRuntime *", "BttValue"]
  @import ["BttRuntime *", "BttImport"]
  @signatures %{
    "query" => ["uint64_t", "BttQuery *"],
    "program" => ["uint32_t", "const char **", "uint32_t *"],
    "entry" => ["BttEntry", "const char **", "uint32_t *"],
    "create" => ["uint64_t", "BttRuntime **"],
    "register" => ["BttRuntime *", "uint64_t"],
    "process" => ["BttRuntime *", "BttEntry", "BttProcess *"],
    "value" => @process ++ ["uint32_t", "const uint8_t *", "uint64_t", "const BttValue *", "uint32_t", "BttValue *"],
    "float" => @process ++ ["uint64_t", "uint64_t", "BttValue *"],
    "float_read" => @value ++ ["uint64_t *"],
    "atom" => @process ++ ["uint64_t", "const uint8_t *", "uint64_t", "BttValue *"],
    "callback" => @process ++ ["uint32_t", "uint32_t", "BttValue *"],
    "export" => @process ++ ["BttEntry", "BttValue *"],
    "call" => @process ++ ["const BttValue *", "uint32_t"],
    "context_set" => @process ++ ["uint64_t", "BttValue"],
    "context_get" => @process ++ ["BttValue *"],
    "step" => ["BttRuntime *", "uint32_t", "BttStep *"],
    "event" => ["BttRuntime *", "BttEvent *"],
    "reply" => @process ++ ["uint64_t", "BttValue", "uint32_t"],
    "result" => @process ++ ["BttValue *", "uint32_t *"],
    "view" => @value ++ ["BttView *"],
    "child" => @value ++ ["uint32_t", "BttValue *"],
    "key" => @value ++ ["uint32_t", "BttValue *"],
    "release" => @value,
    "cancel" => @process,
    "drain" => @process ++ ["uint32_t"],
    "destroy" => ["BttRuntime *"],
    "closure_definition" => ["uint32_t", "uint32_t", "BttCallable *"],
    "closure" => @process ++ ["uint64_t", "uint32_t", "uint32_t", "const BttValue *", "uint32_t", "BttValue *"],
    "opaque" => @process ++ ["uint64_t", "uint32_t", "uint32_t", "BttValue *"],
    "callable_info" => @value ++ ["BttCallable *"],
    "integer_begin" => @process ++ ["uint64_t", "uint64_t", "uint32_t", "BttImport *"],
    "integer_write" => @import ++ ["uint64_t", "const uint8_t *", "uint32_t"],
    "integer_commit" => @import ++ ["BttValue *"],
    "import_cancel" => @import,
    "binary_begin" => @process ++ ["uint64_t", "uint64_t", "BttImport *"],
    "binary_write" => @import ++ ["uint64_t", "const uint8_t *", "uint32_t"],
    "binary_commit" => @import ++ ["BttValue *"],
    "binary_origin" => @import ++ ["uint32_t"],
    "binary_owner" => @value ++ ["BttValue *", "uint64_t *", "uint32_t *", "uint32_t *"],
    "binary_same_owner" => @value ++ ["BttValue", "uint32_t *"],
    "term_begin" => @process ++ ["uint64_t", "uint64_t", "BttImport *"],
    "term_write" => @import ++ ["uint64_t", "const uint8_t *", "uint32_t"],
    "term_bind" => @import ++ ["uint32_t", "BttValue"],
    "term_start" => @import,
    "term_commit" => @import ++ ["BttValue *"],
    "integer_read" => @value ++ ["uint64_t", "uint8_t *", "uint32_t", "uint32_t *", "uint64_t *", "uint32_t *"],
    "copy_begin" => @process ++ ["BttValue", "BttCopy *"],
    "copy_step" => ["BttRuntime *", "BttCopy", "uint32_t", "BttValue *", "uint32_t *"],
    "copy_cancel" => ["BttRuntime *", "BttCopy"]
  }
  @storage %{
    runtime: "!llvm.ptr",
    entry: "!llvm.struct<(i32, i32)>",
    process: "!llvm.struct<(i64, i64, i32, i32)>",
    value: "!llvm.struct<(i64, i64, i32, i32)>",
    copy: "!llvm.struct<(i64, i64, i32, i32)>",
    import: "!llvm.struct<(i64, i64, i64, i32, i32)>",
    callable: "!llvm.struct<(i32, i32, i32, i32, i32, i32, ptr, ptr)>",
    query: "!llvm.struct<(i64, i32, i32, ptr)>",
    step: "!llvm.struct<(i32, i32, i64)>",
    view: "!llvm.struct<(i32, i32, ptr, i64)>",
    event:
      "!llvm.struct<(i64, struct<(i64, i64, i32, i32)>, i32, i32, i32, i32, array<16 x struct<(i64, i64, i32, i32)>>)>"
  }
  @statuses ~w(ok invalid unsupported busy failed waiting done limit cancelled)a |> Enum.with_index() |> Map.new()
  @values %{
    "BttValue" => {"term", :value},
    "BttProcess" => {"process", :process},
    "BttEntry" => {"entry", :entry},
    "BttImport" => {"import_token", :import},
    "BttCopy" => {"copy_job", :copy}
  }
  @outputs Map.merge(Map.new(@values, fn {c, {ex, _}} -> {c <> " *", "!ex." <> ex} end), %{
             "BttRuntime **" => "!llvm.ptr",
             "const char **" => "!llvm.ptr",
             "uint32_t *" => "i32",
             "uint64_t *" => "i64",
             "BttQuery *" => @storage.query,
             "BttStep *" => @storage.step,
             "BttView *" => @storage.view,
             "BttEvent *" => @storage.event,
             "BttCallable *" => @storage.callable
           })
  def abi_id, do: @abi
  def status(name), do: Map.fetch!(@statuses, name)
  def functions, do: @signatures |> Map.keys() |> Enum.sort()
  def functions(layer), do: Enum.filter(functions(), &(layer(&1) == layer))
  def layer(name) when is_map_key(@signatures, name), do: if(name in @execution, do: :execution, else: :runtime)

  def symbol(name, namespace) when is_map_key(@signatures, name),
    do: "beaver_batata_" <> namespace!(namespace) <> "_" <> name

  def argument_types(name), do: Enum.map(Map.fetch!(@signatures, name), &mlir_type/1)
  def function_type(name), do: Type.function(Enum.map(argument_types(name), &Type.get/1), [Type.i32()])
  def storage_type(name, opts \\ []), do: Type.get(Map.fetch!(@storage, name), opts)
  def value_types, do: Map.new(@values, fn {_, {ex, storage}} -> {"!ex." <> ex, Map.fetch!(@storage, storage)} end)

  def value_parameters(name) do
    for type <- Map.fetch!(@signatures, name) do
      cond do
        output = @outputs[type] -> {:out, output}
        value = @values[type] -> {:aggregate, "!ex." <> elem(value, 0)}
        true -> {:in, mlir_type(type)}
      end
    end
  end

  def value_signature(name) do
    params = value_parameters(name)
    {for({direction, type} <- params, direction != :out, do: type), ["i32" | for({:out, type} <- params, do: type)]}
  end

  defp mlir_type("uint32_t"), do: "i32"
  defp mlir_type("uint64_t"), do: "i64"
  defp mlir_type(_), do: "!llvm.ptr"

  @doc "C source to compile with the matching Batata aot.h and admitted library."
  def adapter_source(namespace) do
    namespace = namespace!(namespace)

    """
    #include "aot.h"
    #include <stddef.h>
    _Static_assert(BTT_ABI_ID == UINT64_C(#{@abi}), "unsupported Batata C ABI");
    _Static_assert(sizeof(BttEntry) == 8 && sizeof(BttValue) == 24 && sizeof(BttProcess) == 24, "Batata handle layout");
    _Static_assert(sizeof(BttCopy) == 24 && sizeof(BttImport) == 32, "Batata transaction layout");
    _Static_assert(offsetof(BttCopy, process) == 16 && offsetof(BttImport, process) == 24, "Batata transaction fields");
    _Static_assert(offsetof(BttCallable, program_identity) == 24, "Batata callable layout");
    _Static_assert(offsetof(BttValue, process) == 16 && offsetof(BttProcess, slot) == 16, "Batata handle fields");
    _Static_assert(sizeof(BttStep) == 16 && offsetof(BttStep, reductions) == 8, "Batata step layout");
    _Static_assert(offsetof(BttQuery, identity) == 16 && offsetof(BttView, bytes) == 8, "Batata output layout");
    _Static_assert(sizeof(BttEvent) == 432 && offsetof(BttEvent, arguments) == 48, "Batata event layout");
    #ifdef _WIN32
    #define BEAVER_BATATA_EXPORT __declspec(dllexport)
    #else
    #define BEAVER_BATATA_EXPORT __attribute__((visibility("default")))
    #endif
    """ <> Enum.map_join(functions(), "\n", &adapter(&1, namespace))
  end

  defp namespace!(namespace) do
    unless is_binary(namespace) and byte_size(namespace) in 1..128 and Regex.match?(~r/\A[A-Za-z0-9_]+\z/, namespace),
      do:
        raise(
          ArgumentError,
          "Batata C ABI requires a unique :runtime_namespace of 1-128 ASCII letters, digits or underscores"
        )

    namespace
  end

  defp adapter(name, namespace) do
    params = Map.fetch!(@signatures, name) |> Enum.with_index()

    declaration =
      Enum.map_join(params, ", ", fn {type, i} ->
        "#{if type in @aggregates, do: "const #{type} *", else: type} a#{i}"
      end)

    arguments = Enum.map_join(params, ", ", fn {type, i} -> "#{if type in @aggregates, do: "*"}a#{i}" end)
    checks = for {type, i} <- params, type in @aggregates, do: "if (!a#{i}) return BTT_INVALID;"

    "BEAVER_BATATA_EXPORT int #{symbol(name, namespace)}(#{declaration}) { #{Enum.join(checks, " ")} return btt_#{name}(#{arguments}); }\n"
  end
end
