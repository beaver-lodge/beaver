defmodule Beaver.MLIR.ExecutionEngine do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.Pass.Composer
  import Beaver.MLIR.CAPI

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

  defp do_invoke!(jit, symbol, arg_ptr_list) do
    mlirExecutionEngineInvokePacked(
      jit,
      MLIR.StringRef.create(symbol),
      Beaver.Native.array(arg_ptr_list, Beaver.Native.OpaquePtr, mut: true)
    )
  end

  @doc """
  invoke a function by symbol name.
  """
  def invoke!(jit, symbol, args, return \\ nil) when is_list(args) do
    arg_ptr_list = args |> Enum.map(&Beaver.Native.opaque_ptr/1)

    return_ptr =
      if return do
        return |> Beaver.Native.opaque_ptr()
      else
        []
      end
      |> List.wrap()

    result = do_invoke!(jit, symbol, arg_ptr_list ++ return_ptr)

    if MLIR.LogicalResult.success?(result) do
      return || :ok
    else
      raise "Execution engine invoke failed"
    end
  end

  def destroy(jit) do
    mlirExecutionEngineDestroy(jit)
  end
end
