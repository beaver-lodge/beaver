defmodule Beaver.MLIR.Target.LLVMIR do
  @moduledoc """
  Safe export of fully lowered MLIR modules to textual LLVM IR, native AOT
  object code, concurrent per-function object shards, lazy content-addressed
  multi-unit compilation, and shared library linking.
  """

  alias Beaver.MLIR
  import Bitwise

  @spec translate(MLIR.Module.t()) :: {:ok, binary()} | {:error, MLIR.diagnostics()}
  def translate(%MLIR.Module{} = module) do
    module |> MLIR.context() |> MLIR.Context.register_translations()

    module
    |> MLIR.Operation.from_module()
    |> then(&MLIR.CAPI.beaver_raw_translate_module_to_llvm_ir(&1.ref))
    |> Beaver.Native.check!()
    |> case do
      {llvm_ir, _diagnostics} when is_binary(llvm_ir) -> {:ok, llvm_ir}
      {:error, diagnostics} when is_list(diagnostics) -> {:error, diagnostics}
    end
  end

  @spec translate!(MLIR.Module.t()) :: binary()
  def translate!(%MLIR.Module{} = module) do
    case translate(module) do
      {:ok, llvm_ir} ->
        llvm_ir

      {:error, diagnostics} ->
        raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "failed to translate module to LLVM IR")
    end
  end

  @doc "Returns the host machine's default LLVM target triple, CPU name, and feature string."
  @spec host_target() :: %{triple: binary(), cpu: binary(), features: binary()}
  def host_target() do
    {triple, cpu, features} = MLIR.CAPI.beaver_raw_llvm_host_target()
    %{triple: triple, cpu: cpu, features: features}
  end

  @doc "Returns a normalized `<arch>-<os>` platform identifier for the host target (e.g. `\"aarch64-macos\"`)."
  @spec host_platform() :: String.t()
  def host_platform() do
    %{triple: triple} = host_target()
    parts = String.split(triple, "-")
    arch = normalize_arch(List.first(parts) || "unknown")
    os = detect_os(triple, parts)
    "#{arch}-#{os}"
  end

  defp normalize_arch(arch) when arch in ["arm64", "aarch64"], do: "aarch64"
  defp normalize_arch(arch) when arch in ["x86_64", "amd64"], do: "x86_64"
  defp normalize_arch(arch), do: arch

  defp detect_os(triple, parts) do
    cond do
      String.contains?(triple, ["darwin", "macosx"]) -> "macos"
      String.contains?(triple, "linux") -> "linux"
      String.contains?(triple, ["windows", "msvc"]) -> "windows"
      String.contains?(triple, "freebsd") -> "freebsd"
      true -> Enum.at(parts, 2) || "unknown"
    end
  end

  @doc "Returns the host shared library file extension (`\".dylib\"`, `\".so\"`, or `\".dll\"`)."
  @spec shared_library_extension() :: String.t()
  def shared_library_extension do
    case :os.type() do
      {:unix, :darwin} -> ".dylib"
      {:win32, _} -> ".dll"
      _ -> ".so"
    end
  end

  @doc "Returns a deterministic target configuration descriptor suitable for content-addressed cache keys."
  @spec target_descriptor(keyword()) :: String.t()
  def target_descriptor(opts \\ []) when is_list(opts) do
    host = host_target()
    triple = non_empty(Keyword.get(opts, :triple, ""), host.triple)
    cpu = non_empty(Keyword.get(opts, :cpu, ""), host.cpu)
    features = non_empty(Keyword.get(opts, :features, ""), host.features)
    opt_level = Keyword.get(opts, :opt_level, 2)
    pic = Keyword.get(opts, :pic, true)
    format = Keyword.get(opts, :format, :object)
    "triple=#{triple}\ncpu=#{cpu}\nfeatures=#{features}\nopt=#{opt_level}\npic=#{pic}\nformat=#{format}\n"
  end

  @doc "Compiles an MLIR module (or textual LLVM IR string) to native object code or target assembly in-process."
  @spec compile_object(MLIR.Module.t() | MLIR.Operation.t() | binary(), keyword()) ::
          {:ok, binary()} | {:ok, binary(), map()} | {:error, binary()}
  def compile_object(module_or_llvm_ir, opts \\ []) when is_list(opts) do
    {triple, cpu, features, opt_level, flags, return_metrics} = parse_compile_opts!(opts)
    {module_ref, llvm_ir_arg} = resolve_compile_source!(module_or_llvm_ir)

    case MLIR.CAPI.beaver_raw_compile_to_object(module_ref, llvm_ir_arg, triple, cpu, features, opt_level, flags) do
      {:ok, binary, {translate_ns, opt_ns, codegen_ns}} ->
        if return_metrics,
          do: {:ok, binary, %{translate_ns: translate_ns, opt_ns: opt_ns, codegen_ns: codegen_ns}},
          else: {:ok, binary}

      {:error, message} ->
        {:error, message}
    end
  end

  @doc "Compiles an MLIR module or textual LLVM IR string to object code, raising on failure."
  @spec compile_object!(MLIR.Module.t() | MLIR.Operation.t() | binary(), keyword()) :: binary()
  def compile_object!(module_or_llvm_ir, opts \\ []) do
    case compile_object(module_or_llvm_ir, Keyword.delete(opts, :return_metrics)) do
      {:ok, object} -> object
      {:error, message} -> raise ArgumentError, "failed to compile to object: #{message}"
    end
  end

  @doc "Partitions a lowered MLIR module into per-function compilation units and compiles them concurrently."
  @spec compile_functions(MLIR.Module.t(), keyword()) :: {:ok, [map()]} | {:error, term()}
  def compile_functions(%MLIR.Module{} = module, opts \\ []) when is_list(opts) do
    module |> MLIR.context() |> MLIR.Context.register_translations()
    root_op = MLIR.Operation.from_module(module)
    func_names = Keyword.get_lazy(opts, :functions, fn -> defined_function_names(module) end)
    cache_dir = Keyword.get(opts, :cache_dir)
    if cache_dir, do: File.mkdir_p!(cache_dir)

    compile_opts =
      opts
      |> Keyword.take([:triple, :cpu, :features, :opt_level, :pic, :format])
      |> Keyword.put(:return_metrics, true)

    max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())

    with {:ok, shards} <- extract_shards(root_op, func_names, shard_key_prefix(compile_opts)) do
      compile_shards_concurrently(shards, cache_dir, compile_opts, max_concurrency)
    end
  end

  @doc "Compiles all defined functions in `module` into per-function object shards, raising on failure."
  @spec compile_functions!(MLIR.Module.t(), keyword()) :: [map()]
  def compile_functions!(%MLIR.Module{} = module, opts \\ []) do
    case compile_functions(module, opts) do
      {:ok, shards} -> shards
      {:error, reason} -> raise ArgumentError, "failed to compile module functions: #{inspect(reason)}"
    end
  end

  @doc """
  Compiles lazy compilation units concurrently with two-level content-addressed caching.

  Each unit may be `%{id: id, key: key, source: fun_or_ir, metadata: map()}` or `{id, key, fun_or_ir}`,
  where `source` is a 0-arity `(-> binary() | MLIR.Module.t())` callback, a 1-arity
  `(MLIR.Context.t() -> binary() | MLIR.Module.t())` callback, or an eager binary/module.

  On a verified cache hit under `:cache_dir` (`<cache_dir>/objects/<key>/<key>.o` + `manifest.json`),
  `source` is never invoked and no `MLIR.Context` is created.
  """
  @spec compile_units([map() | tuple()], keyword()) :: {:ok, [map()]} | {:error, term()}
  def compile_units(units, opts \\ []) when is_list(units) and is_list(opts) do
    normalized = Enum.map(units, &normalize_unit!/1)
    cache_dir = Keyword.get(opts, :cache_dir)
    if cache_dir, do: File.mkdir_p!(Path.join(cache_dir, "objects"))
    max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())

    normalized
    |> Task.async_stream(&compile_one_unit(&1, cache_dir, opts),
      max_concurrency: max_concurrency,
      timeout: :infinity
    )
    |> Enum.reduce_while({:ok, []}, fn
      {:ok, {:ok, item}}, {:ok, acc} -> {:cont, {:ok, [item | acc]}}
      {:ok, {:error, err}}, _ -> {:halt, {:error, err}}
      {:exit, reason}, _ -> {:halt, {:error, reason}}
    end)
    |> case do
      {:ok, reversed} -> {:ok, Enum.reverse(reversed)}
      error -> error
    end
  end

  @doc "Compiles lazy compilation units via `compile_units/2`, raising on failure."
  @spec compile_units!([map() | tuple()], keyword()) :: [map()]
  def compile_units!(units, opts \\ []) do
    case compile_units(units, opts) do
      {:ok, compiled} -> compiled
      {:error, reason} -> raise ArgumentError, "failed to compile units: #{inspect(reason)}"
    end
  end

  @doc """
  Links object files (or unit maps returned by `compile_units/2` / `compile_functions/2`)
  into a shared library at `output_path` atomically.

  Options:
    * `:cc` - explicit C compiler/driver executable path.
    * `:ldflags` - extra linker/driver flags (`[String.t()]`, default `[]`).
    * `:libs` - extra library paths or `-l...` flags (`[String.t()]`, default `[]`).
    * `:undefined_lookup` - on macOS, passes `-undefined dynamic_lookup` when `true` (default `true`).
  """
  @spec link_shared([Path.t() | map()], Path.t(), keyword()) :: {:ok, Path.t()} | {:error, binary()}
  def link_shared(object_inputs, output_path, opts \\ [])
      when is_list(object_inputs) and is_binary(output_path) and is_list(opts) do
    if object_inputs == [], do: raise(ArgumentError, "link_shared requires at least one object input")
    expanded_out = Path.expand(output_path)
    out_dir = Path.dirname(expanded_out)
    File.mkdir_p!(out_dir)

    cc = resolve_c_compiler!(opts)
    {obj_paths, temp_objs} = materialize_object_inputs!(object_inputs, out_dir)
    tmp_out = unique_temp_path(expanded_out)

    shared_flags = platform_shared_flags(Keyword.get(opts, :undefined_lookup, true))
    ldflags = Keyword.get(opts, :ldflags, [])
    libs = Keyword.get(opts, :libs, [])
    args = shared_flags ++ obj_paths ++ ldflags ++ libs ++ ["-o", tmp_out]

    try do
      case System.cmd(cc, args, stderr_to_stdout: true) do
        {_out, 0} ->
          File.rename!(tmp_out, expanded_out)
          {:ok, expanded_out}

        {err_output, code} ->
          _ = File.rm(tmp_out)
          {:error, "linker #{cc} exited with status #{code}: #{String.trim(err_output)}"}
      end
    after
      Enum.each(temp_objs, &File.rm/1)
    end
  end

  @doc "Links `object_inputs` into a shared library at `output_path`, raising on failure."
  @spec link_shared!([Path.t() | map()], Path.t(), keyword()) :: Path.t()
  def link_shared!(object_inputs, output_path, opts \\ []) do
    case link_shared(object_inputs, output_path, opts) do
      {:ok, path} -> path
      {:error, reason} -> raise ArgumentError, "failed to link shared library: #{reason}"
    end
  end

  @doc "Compile textual LLVM IR to NVPTX assembly in-process."
  @spec compile_to_ptx(binary(), keyword()) :: {:ok, binary()} | {:error, binary()}
  def compile_to_ptx(llvm_ir, opts \\ []) when is_binary(llvm_ir) and is_list(opts) do
    cpu = Keyword.get(opts, :cpu, "sm_80")
    features = Keyword.get(opts, :features, "")
    unless is_binary(cpu) and is_binary(features), do: raise(ArgumentError, ":cpu and :features must be strings")
    MLIR.CAPI.beaver_raw_compile_llvm_ir_to_ptx(llvm_ir, cpu, features)
  end

  @spec compile_to_ptx!(binary(), keyword()) :: binary()
  def compile_to_ptx!(llvm_ir, opts \\ []) do
    case compile_to_ptx(llvm_ir, opts) do
      {:ok, ptx} -> ptx
      {:error, message} -> raise ArgumentError, "failed to compile LLVM IR to PTX: #{message}"
    end
  end

  defp non_empty("", fallback), do: fallback
  defp non_empty(val, _fallback), do: val

  defp normalize_unit!({id, key, source}),
    do: normalize_unit!(%{id: id, key: key, source: source, metadata: %{}})

  defp normalize_unit!(%{} = map) do
    id = Map.fetch!(map, :id)
    key = Map.fetch!(map, :key) |> to_string()
    source = Map.fetch!(map, :source)
    metadata = Map.get(map, :metadata, %{})

    if key == "" or String.contains?(key, ["/", "\\", ".."]),
      do: raise(ArgumentError, "invalid unit cache key: #{inspect(key)}")

    unless is_map(metadata), do: raise(ArgumentError, "unit :metadata must be a map")
    %{id: id, key: key, source: source, metadata: metadata}
  end

  defp normalize_unit!(other),
    do: raise(ArgumentError, "invalid compilation unit specification: #{inspect(other)}")

  defp compile_one_unit(%{id: id, key: key, source: source, metadata: metadata}, cache_dir, opts) do
    ext = if Keyword.get(opts, :format, :object) == :assembly, do: ".s", else: ".o"
    unit_dir = if cache_dir, do: Path.join([cache_dir, "objects", key]), else: nil
    obj_path = if unit_dir, do: Path.join(unit_dir, key <> ext), else: nil
    manifest_path = if unit_dir, do: Path.join(unit_dir, "manifest.json"), else: nil

    case try_read_unit_cache(obj_path, manifest_path) do
      {:hit, object, sha256, cached_meta} ->
        {:ok,
         %{
           id: id,
           key: key,
           path: obj_path,
           object: object,
           sha256: sha256,
           cached?: true,
           metadata: Map.merge(cached_meta, metadata),
           metrics: %{emit_ns: 0, lower_ns: 0, translate_ns: 0, opt_ns: 0, codegen_ns: 0}
         }}

      :miss ->
        build_and_cache_unit(id, key, source, metadata, obj_path, manifest_path, opts)
    end
  end

  defp try_read_unit_cache(nil, _manifest_path), do: :miss

  defp try_read_unit_cache(obj_path, manifest_path) do
    with true <- File.regular?(obj_path) and File.regular?(manifest_path),
         {:ok, object} <- File.read(obj_path),
         {:ok, manifest_bin} <- File.read(manifest_path),
         manifest when is_map(manifest) <- safe_json_decode(manifest_bin),
         sha256 <- sha256_hex(object),
         true <- manifest["sha256"] == sha256 do
      {:hit, object, sha256, Map.get(manifest, "metadata", %{})}
    else
      _ -> :miss
    end
  end

  defp safe_json_decode(bin) do
    JSON.decode!(bin)
  rescue
    _ -> nil
  end

  defp build_and_cache_unit(id, key, source, metadata, obj_path, manifest_path, opts) do
    compile_opts =
      opts
      |> Keyword.take([:triple, :cpu, :features, :opt_level, :pic, :format])
      |> Keyword.put(:return_metrics, true)

    case produce_unit_object(source, compile_opts, opts) do
      {:ok, object, full_metrics} ->
        sha256 = sha256_hex(object)
        persist_unit_cache(id, key, object, sha256, metadata, obj_path, manifest_path)

        {:ok,
         %{
           id: id,
           key: key,
           path: obj_path,
           object: object,
           sha256: sha256,
           cached?: false,
           metadata: metadata,
           metrics: full_metrics
         }}

      {:error, reason} ->
        {:error, {id, reason}}
    end
  rescue
    e -> {:error, {id, Exception.message(e)}}
  end

  defp produce_unit_object(source, compile_opts, opts) when is_function(source, 1) do
    with_unit_context(opts, fn ctx ->
      {produced, emit_ns} = measure_ns(fn -> source.(ctx) end)
      lower_and_codegen_in_ctx(produced, ctx, emit_ns, compile_opts, opts)
    end)
  end

  defp produce_unit_object(source, compile_opts, opts) when is_function(source, 0) do
    {produced, emit_ns} = measure_ns(source)
    lower_and_codegen_value(produced, emit_ns, compile_opts, opts)
  end

  defp produce_unit_object(source, compile_opts, opts),
    do: lower_and_codegen_value(source, 0, compile_opts, opts)

  defp persist_unit_cache(_id, _key, _object, _sha256, _metadata, nil, _manifest_path), do: :ok
  defp persist_unit_cache(_id, _key, _object, _sha256, _metadata, _obj_path, nil), do: :ok

  defp persist_unit_cache(id, key, object, sha256, metadata, obj_path, manifest_path) do
    write_atomic!(obj_path, object)

    manifest = %{
      "id" => to_string(id),
      "key" => key,
      "sha256" => sha256,
      "size_bytes" => byte_size(object),
      "metadata" => metadata
    }

    write_atomic!(manifest_path, JSON.encode!(manifest))
  end

  defp lower_and_codegen_value(%MLIR.Module{} = mod, emit_ns, compile_opts, opts) do
    ctx = MLIR.context(mod)
    lower_and_codegen_module(mod, ctx, false, emit_ns, compile_opts, opts)
  end

  defp lower_and_codegen_value(text, emit_ns, compile_opts, opts) when is_binary(text) do
    input_kind = Keyword.get(opts, :input, :auto)

    is_mlir? =
      case input_kind do
        :mlir -> true
        :llvm_ir -> false
        :auto -> mlir_text?(text) or Keyword.has_key?(opts, :lower) or Keyword.has_key?(opts, :passes)
      end

    if is_mlir? do
      with_unit_context(opts, fn ctx ->
        mod = MLIR.Module.create!(text, ctx: ctx)
        lower_and_codegen_module(mod, ctx, true, emit_ns, compile_opts, opts)
      end)
    else
      case compile_object(text, compile_opts) do
        {:ok, obj, m} -> {:ok, obj, Map.merge(%{emit_ns: emit_ns, lower_ns: 0}, m)}
        error -> error
      end
    end
  end

  defp lower_and_codegen_in_ctx(%MLIR.Module{} = mod, ctx, emit_ns, compile_opts, opts),
    do: lower_and_codegen_module(mod, ctx, true, emit_ns, compile_opts, opts)

  defp lower_and_codegen_in_ctx(text, ctx, emit_ns, compile_opts, opts) when is_binary(text) do
    if Keyword.get(opts, :input, :auto) == :llvm_ir and not Keyword.has_key?(opts, :lower) and
         not Keyword.has_key?(opts, :passes) do
      case compile_object(text, compile_opts) do
        {:ok, obj, m} -> {:ok, obj, Map.merge(%{emit_ns: emit_ns, lower_ns: 0}, m)}
        error -> error
      end
    else
      mod = MLIR.Module.create!(text, ctx: ctx)
      lower_and_codegen_module(mod, ctx, true, emit_ns, compile_opts, opts)
    end
  end

  defp lower_and_codegen_module(mod, ctx, destroy_mod?, emit_ns, compile_opts, opts) do
    try do
      {lowered, lower_ns} = measure_ns(fn -> run_unit_lowering(mod, ctx, opts) end)

      case compile_object(lowered, compile_opts) do
        {:ok, obj, m} -> {:ok, obj, Map.merge(%{emit_ns: emit_ns, lower_ns: lower_ns}, m)}
        error -> error
      end
    after
      if destroy_mod?, do: MLIR.Module.destroy(mod)
    end
  end

  defp run_unit_lowering(mod, ctx, opts) do
    mod
    |> apply_lower_callback(ctx, Keyword.get(opts, :lower))
    |> apply_unit_passes(Keyword.get(opts, :passes, []))
  end

  defp apply_lower_callback(mod, _ctx, nil), do: mod
  defp apply_lower_callback(mod, _ctx, fun) when is_function(fun, 1), do: fun.(mod)
  defp apply_lower_callback(mod, ctx, fun) when is_function(fun, 2), do: fun.(mod, ctx)

  defp apply_unit_passes(mod, []), do: mod

  defp apply_unit_passes(mod, passes) when is_list(passes) do
    passes
    |> Enum.reduce(Beaver.Composer.new(mod), &Beaver.Composer.append(&2, &1))
    |> Beaver.Composer.run!()
  end

  defp with_unit_context(opts, fun) do
    ctx_opts = Keyword.get(opts, :context_opts, [])
    dialects = Keyword.get(opts, :dialects, [])

    MLIR.Context.with_context(ctx_opts, fn ctx ->
      Enum.each(dialects, &Beaver.Slang.load(ctx, &1))
      fun.(ctx)
    end)
  end

  defp mlir_text?(text) do
    trimmed = String.trim_leading(text)
    String.starts_with?(trimmed, "module") or String.starts_with?(trimmed, "\"builtin.module\"")
  end

  defp measure_ns(fun) do
    t0 = System.monotonic_time(:nanosecond)
    res = fun.()
    {res, System.monotonic_time(:nanosecond) - t0}
  end

  defp sha256_hex(iodata), do: :crypto.hash(:sha256, iodata) |> Base.encode16(case: :lower)

  defp resolve_c_compiler!(opts) do
    candidates = [
      Keyword.get(opts, :cc),
      System.get_env("CC"),
      System.find_executable("cc"),
      System.find_executable("clang"),
      System.find_executable("gcc")
    ]

    Enum.find(candidates, &(is_binary(&1) and &1 != "")) ||
      raise(ArgumentError, "unable to locate a C compiler driver (:cc, $CC, cc, clang, or gcc)")
  end

  defp platform_shared_flags(undefined_lookup?) do
    case :os.type() do
      {:unix, :darwin} ->
        if undefined_lookup?,
          do: ["-dynamiclib", "-undefined", "dynamic_lookup"],
          else: ["-dynamiclib"]

      _ ->
        ["-shared"]
    end
  end

  defp materialize_object_inputs!(inputs, out_dir) do
    Enum.reduce(inputs, {[], []}, fn
      path, {paths, temps} when is_binary(path) ->
        {[Path.expand(path) | paths], temps}

      %{path: path}, {paths, temps} when is_binary(path) ->
        {[Path.expand(path) | paths], temps}

      %{object: bytes}, {paths, temps} when is_binary(bytes) ->
        tmp = unique_temp_path(Path.join(out_dir, "unit.o"))
        File.write!(tmp, bytes)
        {[tmp | paths], [tmp | temps]}

      other, _acc ->
        raise ArgumentError, "unsupported link_shared input: #{inspect(other)}"
    end)
    |> then(fn {paths, temps} -> {Enum.reverse(paths), temps} end)
  end

  defp unique_temp_path(base_path),
    do: "#{base_path}.tmp-#{System.pid()}-#{System.unique_integer([:positive])}"

  defp write_atomic!(path, bytes) do
    path |> Path.dirname() |> File.mkdir_p!()
    tmp = unique_temp_path(path)
    File.write!(tmp, bytes)
    File.rename!(tmp, path)
  end

  defp parse_compile_opts!(opts) do
    triple = Keyword.get(opts, :triple, "")
    cpu = Keyword.get(opts, :cpu, "")
    features = Keyword.get(opts, :features, "")
    opt_level = Keyword.get(opts, :opt_level, 2)
    pic = Keyword.get(opts, :pic, true)
    return_metrics = Keyword.get(opts, :return_metrics, false)

    unless is_binary(triple) and is_binary(cpu) and is_binary(features),
      do: raise(ArgumentError, ":triple, :cpu, and :features must be strings")

    unless opt_level in 0..3 and is_boolean(pic),
      do: raise(ArgumentError, "invalid :opt_level or :pic option")

    emit_asm = parse_format!(Keyword.get(opts, :format, :object))
    flags = if(pic, do: 1, else: 0) ||| if(emit_asm, do: 2, else: 0)
    {triple, cpu, features, opt_level, flags, return_metrics}
  end

  defp parse_format!(:object), do: false
  defp parse_format!(:assembly), do: true
  defp parse_format!(other), do: raise(ArgumentError, ":format must be :object or :assembly, got: #{inspect(other)}")

  defp resolve_compile_source!(%MLIR.Module{} = mod) do
    mod |> MLIR.context() |> MLIR.Context.register_translations()
    {MLIR.Operation.from_module(mod).ref, nil}
  end

  defp resolve_compile_source!(%MLIR.Operation{} = op) do
    op |> MLIR.context() |> MLIR.Context.register_translations()
    {op.ref, nil}
  end

  defp resolve_compile_source!(llvm_ir) when is_binary(llvm_ir), do: {nil, llvm_ir}

  defp resolve_compile_source!(other),
    do: raise(ArgumentError, "expected MLIR.Module, MLIR.Operation, or LLVM IR binary, got: #{inspect(other)}")

  defp shard_key_prefix(opts) do
    [
      "beaver-aot-v1",
      Keyword.get(opts, :triple, ""),
      Keyword.get(opts, :cpu, ""),
      Keyword.get(opts, :features, ""),
      "#{Keyword.get(opts, :opt_level, 2)}",
      "#{Keyword.get(opts, :pic, true)}",
      "#{Keyword.get(opts, :format, :object)}"
    ]
  end

  defp extract_shards(root_op, func_names, key_prefix) do
    func_names
    |> Enum.with_index()
    |> Enum.reduce_while({:ok, []}, fn {name, index}, {:ok, acc} ->
      case extract_one_shard(root_op, to_string(name) |> String.trim_leading("@"), index == 0, key_prefix) do
        {:ok, shard} -> {:cont, {:ok, [shard | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, reversed} -> {:ok, Enum.reverse(reversed)}
      error -> error
    end
  end

  defp extract_one_shard(root_op, name_str, keep_globals, key_prefix) do
    case MLIR.CAPI.beaver_raw_extract_function_module(root_op.ref, name_str, keep_globals) do
      nil ->
        {:error, "function #{name_str} not found or is external"}

      raw_mod ->
        shard_mod = Beaver.Native.check!(raw_mod)
        Beaver.Native.with_resource(shard_mod, &MLIR.Module.destroy/1, &translate_shard(&1, name_str, key_prefix))
    end
  end

  defp translate_shard(shard_mod, name_str, key_prefix) do
    case translate(shard_mod) do
      {:ok, llvm_ir} ->
        digest = sha256_hex([key_prefix, llvm_ir])
        {:ok, {name_str, llvm_ir, digest}}

      {:error, diagnostics} ->
        {:error, MLIR.Diagnostic.format(diagnostics, "failed to translate function #{name_str}")}
    end
  end

  defp compile_shards_concurrently(shards, cache_dir, compile_opts, max_concurrency) do
    shards
    |> Task.async_stream(&compile_one_shard(&1, cache_dir, compile_opts),
      max_concurrency: max_concurrency,
      timeout: :infinity
    )
    |> Enum.reduce_while({:ok, []}, fn
      {:ok, {:ok, item}}, {:ok, acc} -> {:cont, {:ok, [item | acc]}}
      {:ok, {:error, err}}, _ -> {:halt, {:error, err}}
      {:exit, reason}, _ -> {:halt, {:error, reason}}
    end)
    |> case do
      {:ok, items} -> {:ok, Enum.reverse(items)}
      error -> error
    end
  end

  defp compile_one_shard({name_str, llvm_ir, digest}, cache_dir, compile_opts) do
    cache_path = if cache_dir, do: Path.join(cache_dir, digest <> ".o"), else: nil

    if cache_path && File.regular?(cache_path) do
      {:ok,
       %{
         name: name_str,
         object: File.read!(cache_path),
         digest: digest,
         cached?: true,
         metrics: %{translate_ns: 0, opt_ns: 0, codegen_ns: 0}
       }}
    else
      compile_and_cache_shard(name_str, llvm_ir, digest, cache_path, compile_opts)
    end
  end

  defp compile_and_cache_shard(name_str, llvm_ir, digest, cache_path, compile_opts) do
    case compile_object(llvm_ir, compile_opts) do
      {:ok, object, metrics} ->
        if cache_path, do: write_atomic!(cache_path, object)
        {:ok, %{name: name_str, object: object, digest: digest, cached?: false, metrics: metrics}}

      {:error, reason} ->
        {:error, {name_str, reason}}
    end
  end

  defp defined_function_names(%MLIR.Module{} = module) do
    module
    |> MLIR.Module.body()
    |> Beaver.Walker.operations()
    |> Enum.flat_map(&function_symbol_if_defined/1)
  end

  defp function_symbol_if_defined(op) do
    has_blocks = op |> Beaver.Walker.regions() |> Enum.any?(&Enum.any?(Beaver.Walker.blocks(&1)))
    sym_attr = op["sym_name"]

    if MLIR.Operation.name(op) in ["llvm.func", "func.func"] and has_blocks and not MLIR.null?(sym_attr) do
      [sym_attr |> MLIR.to_string() |> String.trim("\"")]
    else
      []
    end
  end
end
