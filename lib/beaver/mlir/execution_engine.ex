alias Beaver.MLIR.{CAPI, StringRef}
alias Beaver.Native

defmodule Beaver.MLIR.ExecutionEngine do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias Beaver.Composer
  import CAPI

  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native

  @doc """
  Create a MLIR JIT engine for a module and check if successful. Usually this module should be of LLVM dialect.
  """
  def create!(composer_or_op, opts \\ [])

  def create!(%Composer{} = composer_or_op, opts), do: Composer.run!(composer_or_op) |> create!(opts)

  @type dirty :: nil | :io_bound | :cpu_bound

  @type opt_level :: 0 | 1 | 2 | 3
  @type shared_lib_path :: String.t()
  @type object_dump :: boolean()
  @type enable_pic :: boolean()
  @type opts :: [
          {:shared_lib_paths, [shared_lib_path]},
          {:symbols, map() | keyword()},
          {:opt_level, opt_level},
          {:object_dump, object_dump},
          {:enable_pic, enable_pic},
          {:debug_info, boolean()},
          {:dirty, dirty},
          {:telemetry, (list(atom()), map(), map() -> any())}
        ]
  @spec create!(MLIR.Module.t(), opts()) :: t()
  def create!(module, opts) do
    shared_lib_paths = Keyword.get(opts, :shared_lib_paths, [])
    symbols = Keyword.get(opts, :symbols, [])
    opt_level = Keyword.get(opts, :opt_level, 2)
    object_dump = Keyword.get(opts, :object_dump, false)
    enable_pic = Keyword.get(opts, :enable_pic, false)
    debug_info = Keyword.get(opts, :debug_info, false)

    module = if debug_info, do: Beaver.MLIR.Debug.attach_llvm_scopes!(module), else: module

    shared_lib_paths_ptr = shared_lib_paths |> Enum.map(&StringRef.create/1) |> Native.array(StringRef)

    {jit, diagnostics} =
      mlirExecutionEngineCreateWithDiagnostics(
        MLIR.context(module),
        module,
        opt_level,
        length(shared_lib_paths),
        shared_lib_paths_ptr,
        object_dump,
        enable_pic
      )

    if MLIR.null?(jit),
      do: raise(ArgumentError, MLIR.Diagnostic.format(diagnostics, "Execution engine creation failed"))

    if symbols != [] and symbols != %{}, do: register_symbols(jit, symbols), else: jit
  end

  @doc """
  invoke a function by symbol name.
  """
  @type invoke_opts :: [{:dirty, dirty}]
  @spec invoke!(t(), String.t() | StringRef.t(), list(), any(), invoke_opts()) :: :ok
  def invoke!(jit, symbol, args \\ [], return \\ nil, opts \\ []) when is_list(args) do
    arg_ptr_list = args |> Enum.map(&Native.opaque_ptr/1)

    return_ptr = if(return, do: return |> Native.opaque_ptr(), else: []) |> List.wrap()

    {result, duration} =
      timed(fn ->
        Native.apply_dirty(
          :mlirExecutionEngineInvokePacked,
          [jit, StringRef.create(symbol), Native.array(arg_ptr_list ++ return_ptr, Native.OpaquePtr, mut: true)],
          opts[:dirty]
        )
      end)

    Beaver.MLIR.Telemetry.emit([:execution], %{duration: duration}, %{symbol: to_string(symbol)}, opts)

    result
    |> then(&if MLIR.LogicalResult.success?(&1), do: return || :ok, else: raise("Execution engine invoke failed"))
  end

  @doc """
  Initialize the JIT runtime.

  If not already initialized, this will be called implicitly when first invocation happens.
  """
  def init(jit), do: tap(jit, &CAPI.mlirExecutionEngineInitialize/1)

  defdelegate destroy(jit), to: CAPI, as: :mlirExecutionEngineDestroy

  @doc "Look up a symbol in an execution engine."
  @spec lookup(t(), String.t() | StringRef.t(), keyword()) :: Native.OpaquePtr.t()
  def lookup(jit, symbol, opts \\ []) do
    function =
      if Keyword.get(opts, :packed, false),
        do: :mlirExecutionEngineLookupPacked,
        else: :mlirExecutionEngineLookup

    apply(CAPI, function, [jit, StringRef.create(symbol)])
  end

  @doc "Register a native pointer under a symbol name."
  @spec register_symbol(t(), String.t() | StringRef.t(), Native.OpaquePtr.t()) :: t()
  def register_symbol(jit, symbol, pointer) do
    clean_symbol = symbol |> to_string() |> String.trim_leading("@")
    mlirExecutionEngineRegisterSymbol(jit, StringRef.create(clean_symbol), pointer)
    jit
  end

  @doc "Looks up a symbol address in a dynamic library or in the current process (`nil`)."
  @spec lookup_library_symbol(Path.t() | nil, String.Chars.t()) :: {:ok, Native.OpaquePtr.t()} | :error
  def lookup_library_symbol(library_path \\ nil, symbol_name) do
    lib_str = if library_path, do: Path.expand(library_path), else: ""
    sym_str = symbol_name |> to_string() |> String.trim_leading("@")
    ptr = beaverDynamicLibrarySearchForAddressOfSymbol(StringRef.create(lib_str), StringRef.create(sym_str))
    if ptr == Native.OpaquePtr.null(), do: :error, else: {:ok, ptr}
  end

  @doc "Registers multiple symbols on `jit` from pointers, `{library_path, native_sym}` tuples, or symbol names."
  @spec register_symbols(t(), map() | keyword()) :: t()
  def register_symbols(jit, symbols) when is_map(symbols) or is_list(symbols) do
    Enum.each(symbols, fn
      {jit_sym, %Native.OpaquePtr{} = ptr} ->
        register_symbol(jit, jit_sym, ptr)

      {jit_sym, {library_path, native_sym}} ->
        case lookup_library_symbol(library_path, native_sym) do
          {:ok, ptr} -> register_symbol(jit, jit_sym, ptr)
          :error -> raise ArgumentError, "symbol #{inspect(native_sym)} not found in #{inspect(library_path)}"
        end

      {jit_sym, native_sym} when is_binary(native_sym) or is_atom(native_sym) ->
        case lookup_library_symbol(nil, native_sym) do
          {:ok, ptr} -> register_symbol(jit, jit_sym, ptr)
          :error -> raise ArgumentError, "symbol #{inspect(native_sym)} not found in loaded dynamic libraries"
        end
    end)

    jit
  end

  @doc "Binds symbols from `library_path` into `jit` with optional aliasing."
  @spec bind_library_symbols!(t(), Path.t(), map() | keyword() | [String.Chars.t()]) :: t()
  def bind_library_symbols!(jit, library_path, symbols) do
    expanded = Path.expand(library_path)

    pairs =
      Enum.map(symbols, fn
        {jit_sym, native_sym} -> {jit_sym, {expanded, native_sym}}
        sym -> {sym, {expanded, sym}}
      end)

    register_symbols(jit, pairs)
  end

  @doc """
  Write the object captured by an engine created with `object_dump: true`.
  """
  @spec emit_object!(t(), Path.t()) :: Path.t()
  def emit_object!(jit, path) do
    path = Path.expand(path)
    path |> Path.dirname() |> File.mkdir_p!()
    mlirExecutionEngineDumpToObjectFile(jit, StringRef.create(path))

    if File.regular?(path),
      do: path,
      else: raise("MLIR execution engine did not emit an object file at #{path}")
  end

  @doc """
  Get the paths to the runtime libraries provided by MLIR.
  """
  def runtime_libs, do: Path.join([:code.priv_dir(:beaver), "lib", "*"]) |> Path.wildcard()

  defp timed(fun) do
    started = System.monotonic_time()
    result = fun.()
    {result, System.monotonic_time() - started}
  end
end
