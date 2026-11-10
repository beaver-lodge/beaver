defmodule Beaver.MLIR.Target.LLVMIR do
  @moduledoc """
  Safe export of fully lowered MLIR modules to textual LLVM IR, native AOT
  object code, and concurrent per-function object shards.
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
        digest = :crypto.hash(:sha256, [key_prefix, llvm_ir]) |> Base.encode16(case: :lower)
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
        if cache_path, do: File.write!(cache_path, object)
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
