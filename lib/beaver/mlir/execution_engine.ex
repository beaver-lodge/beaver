defmodule Beaver.MLIR.ExecutionEngine do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias Beaver.Composer
  import Beaver.MLIR.CAPI

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  @doc """
  Create a MLIR JIT engine for a module and check if successful. Usually this module should be of LLVM dialect.
  """
  def create!(composer_or_op, opts \\ [])

  def create!(%Composer{} = composer_or_op, opts) do
    Composer.run!(composer_or_op) |> create!(opts)
  end

  @type dirty :: nil | :io_bound | :cpu_bound

  @type opt_level :: 0 | 1 | 2 | 3
  @type shared_lib_path :: String.t()
  @type object_dump :: boolean()
  @type enable_pic :: boolean()
  @type opts :: [
          {:shared_lib_paths, [shared_lib_path]},
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
    opt_level = Keyword.get(opts, :opt_level, 2)
    object_dump = Keyword.get(opts, :object_dump, false)
    enable_pic = Keyword.get(opts, :enable_pic, false)
    debug_info = Keyword.get(opts, :debug_info, false)

    module =
      if debug_info do
        Beaver.MLIR.Debug.attach_llvm_scopes!(module)
      else
        module
      end

    shared_lib_paths_ptr =
      shared_lib_paths
      |> Enum.map(&MLIR.StringRef.create/1)
      |> Beaver.Native.array(MLIR.StringRef)

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

    if MLIR.null?(jit) do
      raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "Execution engine creation failed")
    end

    jit
  end

  @doc """
  invoke a function by symbol name.
  """
  @type invoke_opts :: [
          {:dirty, dirty}
        ]
  @spec invoke!(t(), String.t() | MLIR.StringRef.t(), list(), any(), invoke_opts()) :: :ok
  def invoke!(jit, symbol, args \\ [], return \\ nil, opts \\ []) when is_list(args) do
    arg_ptr_list = args |> Enum.map(&Beaver.Native.opaque_ptr/1)

    return_ptr =
      if return do
        return |> Beaver.Native.opaque_ptr()
      else
        []
      end
      |> List.wrap()

    {result, duration} =
      timed(fn ->
        Beaver.Native.apply_dirty(
          :mlirExecutionEngineInvokePacked,
          [
            jit,
            MLIR.StringRef.create(symbol),
            Beaver.Native.array(arg_ptr_list ++ return_ptr, Beaver.Native.OpaquePtr, mut: true)
          ],
          opts[:dirty]
        )
      end)

    Beaver.MLIR.Telemetry.emit(
      [:execution],
      %{duration: duration},
      %{symbol: to_string(symbol)},
      opts
    )

    result
    |> then(
      &if MLIR.LogicalResult.success?(&1) do
        return || :ok
      else
        raise "Execution engine invoke failed"
      end
    )
  end

  @doc """
  Initialize the JIT runtime.

  If not already initialized, this will be called implicitly when first invocation happens.
  """
  def init(jit) do
    tap(jit, &MLIR.CAPI.mlirExecutionEngineInitialize/1)
  end

  defdelegate destroy(jit), to: MLIR.CAPI, as: :mlirExecutionEngineDestroy

  @doc "Look up a symbol in an execution engine."
  @spec lookup(t(), String.t() | MLIR.StringRef.t(), keyword()) :: Beaver.Native.OpaquePtr.t()
  def lookup(jit, symbol, opts \\ []) do
    function =
      if Keyword.get(opts, :packed, false),
        do: :mlirExecutionEngineLookupPacked,
        else: :mlirExecutionEngineLookup

    apply(MLIR.CAPI, function, [jit, MLIR.StringRef.create(symbol)])
  end

  @doc "Register a native pointer under a symbol name."
  @spec register_symbol(t(), String.t() | MLIR.StringRef.t(), Beaver.Native.OpaquePtr.t()) :: t()
  def register_symbol(jit, symbol, pointer) do
    mlirExecutionEngineRegisterSymbol(jit, MLIR.StringRef.create(symbol), pointer)
    jit
  end

  @doc """
  Write the object captured by an engine created with `object_dump: true`.
  """
  @spec emit_object!(t(), Path.t()) :: Path.t()
  def emit_object!(jit, path) do
    path = Path.expand(path)
    path |> Path.dirname() |> File.mkdir_p!()
    mlirExecutionEngineDumpToObjectFile(jit, MLIR.StringRef.create(path))

    if File.regular?(path) do
      path
    else
      raise "MLIR execution engine did not emit an object file at #{path}"
    end
  end

  @doc """
  Get the paths to the runtime libraries provided by MLIR.
  """
  def runtime_libs do
    Path.join([:code.priv_dir(:beaver), "lib", "*"]) |> Path.wildcard()
  end

  defp timed(fun) do
    started = System.monotonic_time()
    result = fun.()
    {result, System.monotonic_time() - started}
  end
end
