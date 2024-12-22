defmodule Beaver.MLIR.ExecutionEngine do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias Beaver.Composer
  import Beaver.MLIR.CAPI

  use Kinda.ResourceKind, forward_module: Beaver.Native

  @doc """
  Create a MLIR JIT engine for a module and check if successful. Usually this module should be of LLVM dialect.
  """
  def create!(%Composer{} = composer_or_op) do
    Composer.run!(composer_or_op) |> create!()
  end

  @type dirty :: nil | :io_bound | :cpu_bound

  @type opt_level :: 0 | 1 | 2 | 3
  @type shared_lib_path :: String.t()
  @type object_dump :: boolean()
  @type opts :: [
          {:shared_lib_paths, [shared_lib_path]},
          {:opt_level, opt_level},
          {:object_dump, object_dump},
          {:dirty, dirty}
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
      Beaver.Native.apply_dirty(
        :mlirExecutionEngineCreate,
        [
          module,
          opt_level,
          length(shared_lib_paths),
          shared_lib_paths_ptr,
          object_dump
        ],
        opts[:dirty]
      )

    if MLIR.null?(jit) do
      raise "Execution engine creation failed"
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

    Beaver.Native.apply_dirty(
      :mlirExecutionEngineInvokePacked,
      [
        jit,
        MLIR.StringRef.create(symbol),
        Beaver.Native.array(arg_ptr_list ++ return_ptr, Beaver.Native.OpaquePtr, mut: true)
      ],
      opts[:dirty]
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
