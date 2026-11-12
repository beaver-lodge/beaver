alias Beaver.MLIR.{Attribute, CAPI, Type}
alias Beaver.Deferred

defmodule Beaver.MLIR.Dialect.LLVM do
  @moduledoc """
  Operations and ABI-oriented builders for the MLIR LLVM dialect.

  The raw generated operation functions remain available. These helpers cover contextual LLVM types, enum
  attributes, symbols, and common function/global/ call shapes that otherwise require textual assembly details.
  """

  alias Beaver.MLIR

  use Beaver.MLIR.Dialect, dialect: "llvm", ops: Beaver.MLIR.Dialect.Registry.ops("llvm")

  @linkages ~w(external available_externally linkonce linkonce_odr weak weak_odr appending internal private extern_weak common)a

  @calling_conventions %{
    c: "ccc",
    fast: "fastcc",
    cold: "coldcc",
    ghc: "cc_10",
    hipe: "cc_11",
    anyreg: "anyregcc",
    preserve_most: "preserve_mostcc",
    preserve_all: "preserve_allcc",
    swift: "swiftcc",
    tail: "tailcc",
    ptx_kernel: "ptx_kernelcc",
    ptx_device: "ptx_devicecc",
    spir_func: "spir_funccc",
    spir_kernel: "spir_kernelcc",
    x86_64_sysv: "x86_64_sysvcc",
    win64: "win64cc",
    amdgpu_kernel: "amdgpu_kernelcc",
    aarch64_vector: "aarch64_vectorcallcc",
    aarch64_sve_vector: "aarch64_sve_vectorcallcc",
    wasm_emscripten_invoke: "wasm_emscripten_invokecc"
  }

  @debug_languages %{c: 0x0002}
  @debug_language_dialects %{simt: 0x01, tile: 0x02}
  @debug_emission_kinds %{none: 0, full: 1, line_tables_only: 2, debug_directives_only: 3}
  @debug_name_table_kinds %{default: 0, gnu: 1, none: 2, apple: 3}

  @doc "Build an opaque LLVM pointer type for an address space."
  def pointer, do: pointer(0, [])
  def pointer(opts) when is_list(opts), do: pointer(0, opts)
  def pointer(address_space) when is_integer(address_space), do: pointer(address_space, [])

  def pointer(address_space, opts) when is_integer(address_space) and address_space >= 0 do
    suffix = if address_space == 0, do: "", else: "<#{address_space}>"
    Type.get("!llvm.ptr#{suffix}", opts)
  end

  @doc "Build an LLVM array type."
  def array(element_type, count, opts \\ []) when is_integer(count) and count >= 0 do
    contextual_type(opts, fn ctx ->
      element_type = render_type(element_type, ctx)
      Type.get("!llvm.array<#{count} x #{element_type}>", ctx: ctx)
    end)
  end

  @doc "Build a literal LLVM struct type."
  def struct(field_types, opts \\ []) when is_list(field_types) do
    contextual_type(opts, fn ctx ->
      fields = Enum.map_join(field_types, ", ", &render_type(&1, ctx))
      packed = if Keyword.get(opts, :packed, false), do: "packed ", else: ""
      Type.get("!llvm.struct<#{packed}(#{fields})>", ctx: ctx)
    end)
  end

  @doc """
  Build an LLVM function type.

  LLVM functions have zero or one result. Set `vararg: true` to append `...` to the parameter list.
  """
  def function_type(params, results, opts \\ []) when is_list(params) and is_list(results) do
    if length(results) > 1, do: raise(ArgumentError, "LLVM function types support at most one result")

    contextual_type(opts, fn ctx ->
      params = Enum.map(params, &render_type(&1, ctx))
      params = if Keyword.get(opts, :vararg, false), do: params ++ ["..."], else: params

      result =
        case results do
          [] -> "void"
          [type] -> render_type(type, ctx)
        end

      Type.get("!llvm.func<#{result} (#{Enum.join(params, ", ")})>", ctx: ctx)
    end)
  end

  @doc "Build a validated `#llvm.linkage` attribute."
  def linkage(value, opts \\ []) when is_atom(value) do
    unless value in @linkages,
      do: raise(ArgumentError, "unsupported LLVM linkage: #{inspect(value)}")

    Attribute.get("#llvm.linkage<#{value}>", opts)
  end

  @doc "Build a validated `#llvm.cconv` attribute for a common calling convention."
  def calling_convention(value, opts \\ []) when is_atom(value) do
    case @calling_conventions do
      %{^value => spelling} -> Attribute.get("#llvm.cconv<#{spelling}>", opts)
      _ -> raise ArgumentError, "unsupported LLVM calling convention: #{inspect(value)}"
    end
  end

  @doc """
  Build an LLVM `DICompileUnit` attribute.

  `:source_language_dialect` is optional and defaults to no dialect. Languages and dialects accept their DWARF
  integer values. The common `:c`, `:simt`, and `:tile` spellings are also accepted.
  """
  def di_compile_unit(opts) when is_list(opts) do
    ctx = Keyword.fetch!(opts, :ctx)
    source_language = debug_enum(Keyword.fetch!(opts, :source_language), @debug_languages)
    dialect = opts[:source_language_dialect]
    source_language_dialect = if is_nil(dialect), do: 0, else: debug_enum(dialect, @debug_language_dialects)
    id = Keyword.get_lazy(opts, :id, fn -> CAPI.mlirDistinctAttrCreate(Attribute.unit(ctx: ctx)) end)
    rec_id = Keyword.get_lazy(opts, :rec_id, fn -> CAPI.mlirDistinctAttrCreate(Attribute.unit(ctx: ctx)) end)

    file =
      Keyword.get_lazy(opts, :file, fn ->
        CAPI.mlirLLVMDIFileAttrGet(
          ctx,
          Attribute.string(Keyword.fetch!(opts, :filename), ctx: ctx),
          Attribute.string(Keyword.get(opts, :directory, ""), ctx: ctx)
        )
      end)

    producer = Attribute.string(Keyword.get(opts, :producer, ""), ctx: ctx)
    split_debug_filename = Attribute.string(Keyword.get(opts, :split_debug_filename, ""), ctx: ctx)
    imported_entities = Keyword.get(opts, :imported_entities, [])
    imported_array = Beaver.Native.array(imported_entities, Attribute)
    emission_kind = debug_enum(Keyword.get(opts, :emission_kind, :full), @debug_emission_kinds)
    name_table_kind = debug_enum(Keyword.get(opts, :name_table_kind, :default), @debug_name_table_kinds)

    CAPI.mlirLLVMDICompileUnitAttrGetWithSourceLanguageDialect(
      ctx,
      rec_id,
      Keyword.get(opts, :rec_self, false),
      id,
      source_language,
      source_language_dialect,
      file,
      producer,
      Keyword.get(opts, :optimized, false),
      emission_kind,
      Keyword.get(opts, :debug_info_for_profiling, false),
      name_table_kind,
      split_debug_filename,
      length(imported_entities),
      imported_array
    )
  end

  @doc "Define an `llvm.func` using the same symbol macro shape as `Func.func`."
  defmacro func(call, opts) do
    quote do
      require Beaver.MLIR.Dialect.Func
      Beaver.MLIR.Dialect.Func.func_like(unquote(call), Beaver.MLIR.Dialect.LLVM.func(), unquote(opts))
    end
  end

  @doc """
  Build an inline-initialized `llvm.mlir.global`.

  Required options are `:sym_name` and `:type`. Optional options include `:value`, `:constant`, `:linkage`,
  `:alignment`, and `:address_space`. Linkage atoms are accepted directly.
  """
  def global(%Beaver.SSA{arguments: arguments, ctx: ctx} = ssa) do
    opts = Keyword.new(arguments)
    sym_name = Keyword.fetch!(opts, :sym_name)
    type = opts |> Keyword.fetch!(:type) |> Deferred.resolve(ctx)
    address_space = Keyword.get(opts, :address_space, 0)

    arguments = [
      global_type: Attribute.type(type),
      sym_name: Attribute.string(to_string(sym_name), ctx: ctx),
      linkage: normalize_linkage(Keyword.get(opts, :linkage, :external), ctx),
      addr_space: Attribute.integer(Type.i32(ctx: ctx), address_space)
    ]

    arguments =
      arguments
      |> maybe_attribute(:value, Keyword.get(opts, :value))
      |> maybe_unit(:constant, Keyword.get(opts, :constant, false), ctx)
      |> maybe_i64(:alignment, Keyword.get(opts, :alignment), ctx)
      |> Kernel.++([fn -> [CAPI.mlirRegionCreate()] end])

    MLIR.Operation.eval_ssa(%Beaver.SSA{ssa | op: mlir_global(), arguments: arguments})
  end

  @doc """
  Build a direct `llvm.call` from operands and a `callee:` symbol.

      LLVM.call_(arg, callee: :identity) >>> Type.i32()
  """
  def call_(%Beaver.SSA{arguments: arguments, ctx: ctx} = ssa) do
    {options, operands} = Enum.split_with(arguments, &match?({key, _} when is_atom(key), &1))
    callee = options |> Keyword.fetch!(:callee) |> to_string()
    options = Keyword.drop(options, [:callee])

    arguments = [
      {:callee_operands, operands},
      {:op_bundle_operands, []},
      {:callee, Attribute.flat_symbol_ref(callee, ctx: ctx)},
      {:operand_segment_sizes, :infer},
      {:op_bundle_sizes, Attribute.dense_array([], Beaver.Native.I32, ctx: ctx)} | options
    ]

    MLIR.Operation.eval_ssa(%Beaver.SSA{ssa | op: call(), arguments: arguments})
  end

  defp contextual_type(opts, fun), do: Deferred.from_opts(opts, fun)

  defp render_type(type, ctx) do
    type
    |> Deferred.resolve(ctx)
    |> case do
      %Type{} = type -> to_string(type)
      other -> raise ArgumentError, "expected an MLIR type, got: #{inspect(other)}"
    end
  end

  defp normalize_linkage(%Attribute{} = linkage, _ctx), do: linkage
  defp normalize_linkage(linkage, ctx) when is_atom(linkage), do: linkage(linkage, ctx: ctx)

  defp maybe_attribute(arguments, _name, nil), do: arguments
  defp maybe_attribute(arguments, name, value), do: arguments ++ [{name, value}]

  defp maybe_unit(arguments, _name, false, _ctx), do: arguments

  defp maybe_unit(arguments, name, true, ctx), do: arguments ++ [{name, Attribute.unit(ctx: ctx)}]

  defp maybe_i64(arguments, _name, nil, _ctx), do: arguments

  defp maybe_i64(arguments, name, value, ctx) when is_integer(value) and value > 0,
    do: arguments ++ [{name, Attribute.integer(Type.i64(ctx: ctx), value)}]

  defp debug_enum(value, _known) when is_integer(value) and value >= 0, do: value

  defp debug_enum(value, known) when is_atom(value) do
    case known do
      %{^value => encoded} -> encoded
      _ -> raise ArgumentError, "unsupported LLVM debug enum value: #{inspect(value)}"
    end
  end
end
