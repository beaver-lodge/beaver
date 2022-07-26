defmodule Beaver.MLIR.ExecutionEngine do
  alias Beaver.MLIR
  alias Beaver.MLIR.Pass.Composer
  alias Beaver.MLIR.CAPI
  import Beaver.MLIR.CAPI

  def is_null(jit) do
    jit
    |> beaverMlirExecutionEngineIsNull()
    |> to_term()
  end

  @doc """
  Create a MLIR JIT engine for a module and check if successful. Usually this module should be of LLVM dialect.
  """
  def create!(composer_or_op = %Composer{}) do
    Composer.run!(composer_or_op) |> create!()
  end

  def create!(module, opts \\ []) do
    shared_lib_paths = Keyword.get(opts, :shared_lib_paths, [])

    shared_lib_paths_ptr =
      case shared_lib_paths do
        [] ->
          MLIR.CAPI.array([], MLIR.CAPI.MlirStringRef)

        _ ->
          shared_lib_paths
          |> Enum.map(&MLIR.StringRef.create/1)
          |> MLIR.CAPI.array(MLIR.CAPI.MlirStringRef)
      end

    ctx =
      MLIR.CAPI.mlirModuleGetOperation(module)
      |> MLIR.CAPI.mlirOperationGetContext()

    require MLIR.Context

    MLIR.Context.allow_multi_thread ctx do
      jit =
        mlirExecutionEngineCreate(
          module,
          2,
          length(shared_lib_paths),
          shared_lib_paths_ptr
        )
    end

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
      CAPI.OpaquePtr.array(arg_ptr_list, mut: true)
    )
  end

  @doc """
  invoke a function by symbol name.
  """
  def invoke!(jit, symbol, args, return) when is_list(args) do
    arg_ptr_list = args |> Enum.map(&CAPI.opaque_ptr/1)
    return_ptr = return |> CAPI.opaque_ptr()
    result = do_invoke!(jit, symbol, arg_ptr_list ++ [return_ptr])

    if MLIR.LogicalResult.success?(result) do
      return
    else
      raise "Execution engine invoke failed"
    end
  end

  @doc """
  invoke a void function by symbol name.
  """
  def invoke!(jit, symbol, args) when is_list(args) do
    arg_ptr_list = args |> Enum.map(&CAPI.opaque_ptr/1)
    result = do_invoke!(jit, symbol, arg_ptr_list)

    if MLIR.LogicalResult.success?(result) do
      :ok
    else
      raise "Execution engine invoke failed"
    end
  end

  def destroy(jit) do
    mlirExecutionEngineDestroy(jit)
  end
end
