defmodule Beaver.MLIR.Target.LLVMIR do
  @moduledoc "MLIR export, native AOT compilation, concurrent cached units and atomic shared-library linking."
  alias Beaver.{Composer, MLIR, Native, Slang, Walker}
  alias MLIR.{CAPI, Context, Diagnostic, Module, Operation}
  import Bitwise
  @compile_keys ~w(triple cpu features opt_level pic format)a
  @native_metrics %{translate_ns: 0, opt_ns: 0, codegen_ns: 0}
  @unit_metrics Map.merge(@native_metrics, %{emit_ns: 0, lower_ns: 0})
  @spec translate(Module.t()) :: {:ok, binary()} | {:error, MLIR.diagnostics()}
  def translate(%Module{} = module) do
    Context.register_translations(MLIR.context(module))

    case CAPI.beaver_raw_translate_module_to_llvm_ir(Operation.from_module(module).ref) |> Native.check!() do
      {ir, _} when is_binary(ir) -> {:ok, ir}
      {:error, diagnostics} -> {:error, diagnostics}
    end
  end

  @spec translate!(Module.t()) :: binary()
  def translate!(module) do
    case translate(module) do
      {:ok, ir} -> ir
      {:error, ds} -> raise ArgumentError, Diagnostic.format(ds, "failed to translate module to LLVM IR")
    end
  end

  @doc "Returns the LLVM host triple, CPU name and feature string."
  @spec host_target() :: %{triple: binary(), cpu: binary(), features: binary()}
  def host_target do
    {triple, cpu, features} = CAPI.beaver_raw_llvm_host_target()
    %{triple: triple, cpu: cpu, features: features}
  end

  @doc "Returns the normalized host platform, e.g. `aarch64-macos`."
  @spec host_platform() :: String.t()
  def host_platform do
    triple = host_target().triple
    parts = String.split(triple, "-")
    arch = hd(parts)
    arch = %{"arm64" => "aarch64", "amd64" => "x86_64"}[arch] || arch

    os =
      Enum.find_value(
        [{"macos", ~w(darwin macosx)}, {"linux", ["linux"]}, {"windows", ~w(windows msvc)}, {"freebsd", ["freebsd"]}],
        fn {os, names} -> if String.contains?(triple, names), do: os end
      ) || Enum.at(parts, 2) || "unknown"

    "#{arch}-#{os}"
  end

  @doc "Returns `.dylib`, `.dll` or `.so` for the host OS."
  @spec shared_library_extension() :: String.t()
  def shared_library_extension do
    case :os.type() do
      {:unix, :darwin} -> ".dylib"
      {:win32, _} -> ".dll"
      _ -> ".so"
    end
  end

  @doc "Returns a deterministic descriptor of target, optimization, PIC and output-format settings."
  @spec target_descriptor(keyword()) :: String.t()
  def target_descriptor(opts \\ []) do
    host = host_target()

    target =
      for k <- ~w(triple cpu features)a do
        v = Keyword.get(opts, k, "")
        "#{k}=#{if v == "", do: host[k], else: v}\n"
      end

    IO.iodata_to_binary(
      target ++
        [
          "opt=#{Keyword.get(opts, :opt_level, 2)}\npic=#{Keyword.get(opts, :pic, true)}\nformat=#{Keyword.get(opts, :format, :object)}\n"
        ]
    )
  end

  @doc "Compiles MLIR or textual LLVM IR to an object (or `format: :assembly`), optionally returning timing metrics."
  @spec compile_object(Module.t() | Operation.t() | binary(), keyword()) ::
          {:ok, binary()} | {:ok, binary(), map()} | {:error, binary()}
  def compile_object(source, opts \\ []) do
    {triple, cpu, features, level, flags, metrics?} = compile_options!(opts)
    {module, ir} = compile_source!(source)

    case CAPI.beaver_raw_compile_to_object(module, ir, triple, cpu, features, level, flags) do
      {:ok, bytes, {translate, optimize, codegen}} ->
        metrics = %{translate_ns: translate, opt_ns: optimize, codegen_ns: codegen}
        if metrics?, do: {:ok, bytes, metrics}, else: {:ok, bytes}

      error ->
        error
    end
  end

  @doc "Compiles an object, raising on failure."
  @spec compile_object!(Module.t() | Operation.t() | binary(), keyword()) :: binary()
  def compile_object!(source, opts \\ []),
    do: unwrap!(compile_object(source, Keyword.delete(opts, :return_metrics)), "compile to object")

  @doc "Compiles textual LLVM IR to NVPTX assembly in-process."
  @spec compile_to_ptx(binary(), keyword()) :: {:ok, binary()} | {:error, binary()}
  def compile_to_ptx(ir, opts \\ []) when is_binary(ir) do
    cpu = Keyword.get(opts, :cpu, "sm_80")
    features = Keyword.get(opts, :features, "")
    unless is_binary(cpu) and is_binary(features), do: raise(ArgumentError, ":cpu and :features must be strings")
    CAPI.beaver_raw_compile_llvm_ir_to_ptx(ir, cpu, features)
  end

  @spec compile_to_ptx!(binary(), keyword()) :: binary()
  def compile_to_ptx!(ir, opts \\ []), do: unwrap!(compile_to_ptx(ir, opts), "compile LLVM IR to PTX")
  @doc "Partitions a lowered module into concurrent, optionally cached per-function object shards."
  @spec compile_functions(Module.t(), keyword()) :: {:ok, [map()]} | {:error, term()}
  def compile_functions(%Module{} = module, opts \\ []) do
    Context.register_translations(MLIR.context(module))
    root = Operation.from_module(module)
    names = Keyword.get_lazy(opts, :functions, fn -> defined_functions(module) end)
    options = Keyword.take(opts, [:cache_dir, :max_concurrency | @compile_keys]) |> Keyword.put(:input, :llvm_ir)
    prefix = ["beaver-aot-v1" | Enum.map(~w(triple cpu features)a, &Keyword.get(opts, &1, ""))]

    prefix =
      prefix ++ Enum.map([opt_level: 2, pic: true, format: :object], fn {k, v} -> "#{Keyword.get(opts, k, v)}" end)

    with {:ok, shards} <-
           gather(
             Stream.map(Enum.with_index(names), fn {name, index} ->
               extract_shard(root, name |> to_string() |> String.trim_leading("@"), index == 0, prefix)
             end)
           ),
         {:ok, compiled} <- compile_units(shards, options) do
      results =
        for unit <- compiled do
          Map.take(unit, [:object, :cached?])
          |> Map.merge(%{name: unit.id, digest: unit.key, metrics: Map.take(unit.metrics, Map.keys(@native_metrics))})
        end

      {:ok, results}
    end
  end

  @spec compile_functions!(Module.t(), keyword()) :: [map()]
  def compile_functions!(module, opts \\ []),
    do: unwrap!(compile_functions(module, opts), "compile module functions", true)

  @doc """
  Compiles `{id, key, source}` or `%{id: id, key: key, source: source, metadata: map}` units concurrently. Sources
  can be eager MLIR modules/text, LLVM IR, or callbacks of arity 0 or 1 (receiving an owned context). Verified hits
  at `<cache_dir>/objects/<key>/<key>.o` plus `manifest.json` skip both source and context creation. `:input`
  selects `:mlir`, `:llvm_ir` or `:auto`; `:lower` and `:passes` run before code generation. `:context_opts`,
  `:dialects`, `:max_concurrency` and the `compile_object/2` options are forwarded.
  """
  @spec compile_units([map() | tuple()], keyword()) :: {:ok, [map()]} | {:error, term()}
  def compile_units(units, opts \\ []) do
    units = Enum.map(units, &unit!/1)
    if opts[:cache_dir], do: File.mkdir_p!(Path.join(opts[:cache_dir], "objects"))
    concurrent(units, &compile_unit(&1, opts), opts)
  end

  @spec compile_units!([map() | tuple()], keyword()) :: [map()]
  def compile_units!(units, opts \\ []), do: unwrap!(compile_units(units, opts), "compile units", true)
  defp unit!({id, key, source}), do: unit!(%{id: id, key: key, source: source})

  defp unit!(%{id: id, key: key, source: source} = unit) do
    key = to_string(key)

    if key == "" or String.contains?(key, ["/", "\\", ".."]),
      do: raise(ArgumentError, "invalid unit cache key: #{inspect(key)}")

    metadata = Map.get(unit, :metadata, %{})
    unless is_map(metadata), do: raise(ArgumentError, "unit :metadata must be a map")
    %{id: id, key: key, source: source, metadata: metadata}
  end

  defp unit!(other), do: raise(ArgumentError, "invalid compilation unit specification: #{inspect(other)}")

  defp compile_unit(unit, opts) do
    ext = if opts[:format] == :assembly, do: ".s", else: ".o"
    dir = if opts[:cache_dir], do: Path.join([opts[:cache_dir], "objects", unit.key])
    path = if dir, do: Path.join(dir, unit.key <> ext)
    manifest = if dir, do: Path.join(dir, "manifest.json")
    base = Map.take(unit, [:id, :key, :metadata]) |> Map.put(:path, path)

    case read_unit_cache(path, manifest) do
      {:ok, bytes, metadata} -> unit_result(base, bytes, true, @unit_metrics, Map.merge(metadata, unit.metadata))
      :miss -> build_unit(unit, base, manifest, opts)
    end
  end

  defp read_unit_cache(nil, _), do: :miss

  defp read_unit_cache(path, manifest) do
    with {:ok, bytes} <- File.read(path),
         {:ok, json} <- File.read(manifest),
         {:ok, %{"sha256" => digest} = data} <- JSON.decode(json),
         true <- digest == sha256(bytes),
         metadata when is_map(metadata) <- Map.get(data, "metadata", %{}) do
      {:ok, bytes, metadata}
    else
      _ -> :miss
    end
  end

  defp build_unit(unit, base, manifest, opts) do
    case produce(unit.source, opts) do
      {:ok, bytes, metrics} ->
        if base.path do
          write_atomic!(base.path, bytes)

          data = %{
            id: to_string(unit.id),
            key: unit.key,
            sha256: sha256(bytes),
            size_bytes: byte_size(bytes),
            metadata: unit.metadata
          }

          write_atomic!(manifest, JSON.encode!(data))
        end

        unit_result(base, bytes, false, metrics, unit.metadata)

      {:error, reason} ->
        {:error, {unit.id, reason}}
    end
  rescue
    e -> {:error, {unit.id, Exception.message(e)}}
  end

  defp unit_result(base, bytes, cached?, metrics, metadata),
    do:
      {:ok,
       Map.merge(base, %{object: bytes, sha256: sha256(bytes), cached?: cached?, metrics: metrics, metadata: metadata})}

  defp produce(source, opts) when is_function(source, 1) do
    with_context(opts, fn ctx ->
      {value, elapsed} = measure(fn -> source.(ctx) end)
      codegen(value, ctx, elapsed, opts)
    end)
  end

  defp produce(source, opts) when is_function(source, 0) do
    {value, elapsed} = measure(source)
    codegen(value, nil, elapsed, opts)
  end

  defp produce(source, opts), do: codegen(source, nil, 0, opts)
  # A module returned by a context callback is owned; eager modules and zero-arity callback results are borrowed.
  defp codegen(%Module{} = module, ctx, emit_ns, opts) do
    try do
      {lowered, lower_ns} = measure(fn -> lower(module, ctx || MLIR.context(module), opts) end)
      compile_with_metrics(lowered, %{emit_ns: emit_ns, lower_ns: lower_ns}, opts)
    after
      if ctx, do: Module.destroy(module)
    end
  end

  defp codegen(text, ctx, emit_ns, opts) when is_binary(text) do
    if mlir_source?(text, ctx, opts) do
      if ctx,
        do: codegen(Module.create!(text, ctx: ctx), ctx, emit_ns, opts),
        else: with_context(opts, &codegen(Module.create!(text, ctx: &1), &1, emit_ns, opts))
    else
      compile_with_metrics(text, %{emit_ns: emit_ns, lower_ns: 0}, opts)
    end
  end

  defp mlir_source?(text, nil, opts) do
    case Keyword.get(opts, :input, :auto) do
      :mlir -> true
      :llvm_ir -> false
      :auto -> String.starts_with?(String.trim_leading(text), ["module", "\"builtin.module\""]) or lowering?(opts)
    end
  end

  defp mlir_source?(_, _, opts), do: Keyword.get(opts, :input, :auto) != :llvm_ir or lowering?(opts)
  defp lowering?(opts), do: Keyword.has_key?(opts, :lower) or Keyword.has_key?(opts, :passes)

  defp compile_with_metrics(source, metrics, opts) do
    case compile_object(source, Keyword.put(Keyword.take(opts, @compile_keys), :return_metrics, true)) do
      {:ok, bytes, native} -> {:ok, bytes, Map.merge(metrics, native)}
      error -> error
    end
  end

  defp lower(module, ctx, opts) do
    module =
      case opts[:lower] do
        nil -> module
        f when is_function(f, 1) -> f.(module)
        f when is_function(f, 2) -> f.(module, ctx)
      end

    case Keyword.get(opts, :passes, []) do
      [] -> module
      passes -> Enum.reduce(passes, Composer.new(module), &Composer.append(&2, &1)) |> Composer.run!()
    end
  end

  defp with_context(opts, fun) do
    Context.with_context(Keyword.get(opts, :context_opts, []), fn ctx ->
      Enum.each(Keyword.get(opts, :dialects, []), &Slang.load(ctx, &1))
      fun.(ctx)
    end)
  end

  @doc """
  Atomically links file paths or compiled unit/shard maps into `output_path`. Options: `:cc` (compiler driver),
  `:ldflags`, `:libs`, and `:undefined_lookup` (macOS, default true).
  """
  @spec link_shared([Path.t() | map()], Path.t(), keyword()) :: {:ok, Path.t()} | {:error, binary()}
  def link_shared(inputs, output, opts \\ []) do
    if inputs == [], do: raise(ArgumentError, "link_shared requires at least one object input")
    output = Path.expand(output)
    dir = Path.dirname(output)
    File.mkdir_p!(dir)

    cc =
      Enum.find(
        [opts[:cc], System.get_env("CC") | Enum.map(~w(cc clang gcc), &System.find_executable/1)],
        &(is_binary(&1) and &1 != "")
      ) || raise(ArgumentError, "unable to locate a C compiler driver (:cc, $CC, cc, clang, or gcc)")

    {paths, temps} = Enum.map_reduce(inputs, [], &object_path(&1, &2, dir))
    tmp = temp_path(output)

    flags =
      if :os.type() == {:unix, :darwin},
        do:
          ["-dynamiclib"] ++
            if(Keyword.get(opts, :undefined_lookup, true), do: ["-undefined", "dynamic_lookup"], else: []),
        else: ["-shared"]

    try do
      args = flags ++ paths ++ Keyword.get(opts, :ldflags, []) ++ Keyword.get(opts, :libs, []) ++ ["-o", tmp]

      case System.cmd(cc, args, stderr_to_stdout: true) do
        {_, 0} ->
          File.rename!(tmp, output)
          {:ok, output}

        {message, code} ->
          {:error, "linker #{cc} exited with status #{code}: #{String.trim(message)}"}
      end
    after
      Enum.each([tmp | temps], &File.rm/1)
    end
  end

  @spec link_shared!([Path.t() | map()], Path.t(), keyword()) :: Path.t()
  def link_shared!(inputs, output, opts \\ []), do: unwrap!(link_shared(inputs, output, opts), "link shared library")
  defp object_path(%{path: path}, temps, dir) when is_binary(path), do: object_path(path, temps, dir)
  defp object_path(path, temps, _) when is_binary(path), do: {Path.expand(path), temps}

  defp object_path(%{object: bytes}, temps, dir) when is_binary(bytes) do
    path = temp_path(Path.join(dir, "unit.o"))
    File.write!(path, bytes)
    {path, [path | temps]}
  end

  defp object_path(other, _, _), do: raise(ArgumentError, "unsupported link_shared input: #{inspect(other)}")

  defp compile_options!(opts) do
    [triple, cpu, features] = Enum.map(~w(triple cpu features)a, &Keyword.get(opts, &1, ""))
    level = Keyword.get(opts, :opt_level, 2)
    pic = Keyword.get(opts, :pic, true)

    unless Enum.all?([triple, cpu, features], &is_binary/1),
      do: raise(ArgumentError, ":triple, :cpu, and :features must be strings")

    unless level in 0..3 and is_boolean(pic), do: raise(ArgumentError, "invalid :opt_level or :pic option")
    format = Keyword.get(opts, :format, :object)

    unless format in [:object, :assembly],
      do: raise(ArgumentError, ":format must be :object or :assembly, got: #{inspect(format)}")

    {triple, cpu, features, level, if(pic, do: 1, else: 0) ||| if(format == :assembly, do: 2, else: 0),
     opts[:return_metrics]}
  end

  defp compile_source!(%Module{} = module), do: compile_source!(Operation.from_module(module))

  defp compile_source!(%Operation{} = op) do
    Context.register_translations(MLIR.context(op))
    {op.ref, nil}
  end

  defp compile_source!(text) when is_binary(text), do: {nil, text}

  defp compile_source!(other),
    do: raise(ArgumentError, "expected MLIR.Module, MLIR.Operation, or LLVM IR binary, got: #{inspect(other)}")

  defp extract_shard(root, name, globals?, prefix) do
    case CAPI.beaver_raw_extract_function_module(root.ref, name, globals?) do
      nil ->
        {:error, "function #{name} not found or is external"}

      raw ->
        Module.with_module(Native.check!(raw), &translate_shard(&1, name, prefix))
    end
  end

  defp translate_shard(module, name, prefix) do
    case translate(module) do
      {:ok, ir} -> {:ok, {name, sha256([prefix, ir]), ir}}
      {:error, ds} -> {:error, Diagnostic.format(ds, "failed to translate function #{name}")}
    end
  end

  defp defined_functions(module) do
    for op <- Walker.operations(Module.body(module)),
        Operation.name(op) in ~w(llvm.func func.func),
        Enum.any?(Walker.regions(op), &Enum.any?(Walker.blocks(&1))),
        attr = op["sym_name"],
        not MLIR.null?(attr),
        do: attr |> MLIR.to_string() |> String.trim("\"")
  end

  defp concurrent(items, fun, opts) do
    Task.async_stream(items, fun,
      max_concurrency: Keyword.get(opts, :max_concurrency, System.schedulers_online()),
      timeout: :infinity
    )
    |> Stream.map(fn
      {:ok, result} -> result
      {:exit, reason} -> {:error, reason}
    end)
    |> gather()
  end

  defp gather(results) do
    Enum.reduce_while(results, {:ok, []}, fn
      {:ok, item}, {:ok, acc} -> {:cont, {:ok, [item | acc]}}
      {:error, _} = error, _ -> {:halt, error}
    end)
    |> case do
      {:ok, items} -> {:ok, Enum.reverse(items)}
      error -> error
    end
  end

  defp unwrap!(result, label, inspect? \\ false)
  defp unwrap!({:ok, value}, _label, _inspect?), do: value

  defp unwrap!({:error, reason}, label, inspect?),
    do: raise(ArgumentError, "failed to #{label}: #{if inspect?, do: inspect(reason), else: reason}")

  defp measure(fun) do
    start = System.monotonic_time(:nanosecond)
    value = fun.()
    {value, System.monotonic_time(:nanosecond) - start}
  end

  defp sha256(bytes), do: Beaver.Build.ArtifactCache.digest(bytes)
  defp temp_path(path), do: "#{path}.tmp-#{System.pid()}-#{System.unique_integer([:positive])}"
  defp write_atomic!(path, bytes), do: Beaver.Build.ArtifactCache.write_atomic!(path, bytes)
end
