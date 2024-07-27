defmodule Beaver.MLIR.ExecutionEngine do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.Pass.Composer
  import Beaver.MLIR.CAPI

  use Kinda.ResourceKind,
    root_module: Beaver.MLIR.CAPI,
    forward_module: Beaver.Native

  def is_null(jit) do
    jit
    |> beaverMlirExecutionEngineIsNull()
    |> Beaver.Native.to_term()
  end

  @doc """
  Create a MLIR JIT engine for a module and check if successful. Usually this module should be of LLVM dialect.
  """
  def create!(%Composer{} = composer_or_op) do
    Composer.run!(composer_or_op) |> create!()
  end

  @type opt_level :: 0 | 1 | 2 | 3
  @type shared_lib_path :: String.t()
  @type object_dump :: boolean()
  @type opts :: [
          {:shared_lib_paths, [shared_lib_path]},
          {:opt_level, opt_level},
          {:object_dump, object_dump}
        ]
  @spec create!(MLIR.Module.t(), opts()) :: t()
  def create!(module, opts \\ []) do
    shared_lib_paths = Keyword.get(opts, :shared_lib_paths, [])
    opt_level = Keyword.get(opts, :opt_level, 2)
    object_dump = Keyword.get(opts, :object_dump, false)

    shared_lib_paths_ptr =
      shared_lib_paths
      |> Enum.map(&MLIR.StringRef.create/1)
      |> Beaver.Native.array(MLIR.StringRef)

    require MLIR.Context

    jit =
      mlirExecutionEngineCreate(
        module,
        opt_level,
        length(shared_lib_paths),
        shared_lib_paths_ptr,
        object_dump
      )

    is_null = is_null(jit)

    if is_null do
      raise "Execution engine creation failed"
    end

    jit
  end

  @doc """
  invoke a function by symbol name.
  """
  @type dirty :: nil | :io_bound | :cpu_bound
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

    case opts[:dirty] do
      :io_bound ->
        :mlirExecutionEngineInvokePacked_dirty_io

      :cpu_bound ->
        :mlirExecutionEngineInvokePacked_dirty_cpu

      nil ->
        :mlirExecutionEngineInvokePacked
    end
    |> then(
      &apply(MLIR.CAPI, &1, [
        jit,
        MLIR.StringRef.create(symbol),
        Beaver.Native.array(arg_ptr_list ++ return_ptr, Beaver.Native.OpaquePtr, mut: true)
      ])
    )
    |> then(
      &if MLIR.LogicalResult.success?(&1) do
        return || :ok
      else
        raise "Execution engine invoke failed"
      end
    )
  end

  def destroy(jit) do
    mlirExecutionEngineDestroy(jit)
  end
end
