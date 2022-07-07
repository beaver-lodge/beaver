defmodule Beaver.MLIR.ExecutionEngine do
  alias Beaver.MLIR
  alias Beaver.MLIR.Pass.Composer
  import Beaver.MLIR.CAPI

  def is_null(jit) do
    jit
    |> Exotic.Value.fetch(MLIR.CAPI.MlirExecutionEngine, :ptr)
    |> Exotic.Value.extract() == 0
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
      shared_lib_paths
      |> Enum.map(&MLIR.StringRef.create/1)
      |> Exotic.Value.Array.get()
      |> Exotic.Value.get_ptr()

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
    sym = MLIR.StringRef.create(symbol)

    args_ptr =
      arg_ptr_list
      |> Exotic.Value.Array.get()
      |> Exotic.Value.get_ptr()

    Exotic.call!(MLIR.CAPI, :mlirExecutionEngineInvokePacked, [jit, sym, args_ptr], dirty: false)
    # mlirExecutionEngineInvokePacked(jit, sym, args_ptr)
  end

  @doc """
  invoke a function by symbol name. The arguments should be a list of Exotic.Valuable
  """
  def invoke!(jit, symbol, args, return) when is_list(args) do
    arg_ptr_list = args |> Enum.map(&Exotic.Value.get_ptr/1)
    return_ptr = return |> Exotic.Value.get_ptr()
    result = do_invoke!(jit, symbol, arg_ptr_list ++ [return_ptr])

    if MLIR.LogicalResult.success?(result) do
      return
    else
      raise "Execution engine invoke failed"
    end
  end

  @doc """
  invoke a void function by symbol name. The arguments should be a list of Exotic.Valuable
  """
  def invoke!(jit, symbol, args) when is_list(args) do
    arg_ptr_list = args |> Enum.map(&Exotic.Value.get_ptr/1)
    result = do_invoke!(jit, symbol, arg_ptr_list)

    if not MLIR.LogicalResult.success?(result) do
      raise "Execution engine invoke failed"
    end
  end

  def destroy(jit) do
    mlirExecutionEngineDestroy(jit)
  end
end
