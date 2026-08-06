defmodule Beaver.MLIR.Context do
  @moduledoc """
  This module defines functions creating or destroying MLIR context.
  """
  alias Beaver.MLIR
  import MLIR.CAPI

  use Kinda.ResourceKind,
    raw_module: Beaver.MLIR.CAPI.Raw,
    codec: Beaver.Native,
    fields: [thread_pool_owner: nil, thread_pool_lease: nil]

  @doc """
  Run a function with a registry appended to the context.
  """
  def with_registry(ctx, fun) when is_function(fun, 1) do
    registry = mlirDialectRegistryCreate()
    mlirContextAppendDialectRegistry(ctx, registry)

    try do
      fun.(registry)
    after
      mlirDialectRegistryDestroy(registry)
    end
  end

  # create an interim registry and append all dialects to the context
  defp load_all_dialects(ctx) do
    with_registry(ctx, fn registry ->
      mlirRegisterAllDialects(registry)
      # Appending copies the registry's current contents. `with_registry/2`
      # appended it while empty, so append again after registration.
      mlirContextAppendDialectRegistry(ctx, registry)
      mlirContextLoadAllAvailableDialects(ctx)
    end)
  end

  @type context_option ::
          {:allow_unregistered, boolean()}
          | {:all_dialects, boolean()}
          | {:threading, boolean()}
          | {:thread_pool, :application | MLIR.LLVMThreadPool.t() | GenServer.server() | nil}
          | {:registry, MLIR.DialectRegistry.t() | [MLIR.DialectRegistry.t()]}
  @spec create([context_option()]) :: __MODULE__.t()
  @doc """
  Create a MLIR context. By default it registers and loads all dialects.
  """
  def create(opts \\ []) do
    allow_unregistered = Keyword.get(opts, :allow_unregistered, false)
    all_dialects = Keyword.get(opts, :all_dialects, true)
    threading = Keyword.get(opts, :threading, true)

    if not threading and Keyword.has_key?(opts, :thread_pool) do
      raise ArgumentError, ":thread_pool cannot be used with threading: false"
    end

    {pool, owner, lease} = resolve_thread_pool(opts, threading)
    ctx = mlirContextCreate()

    try do
      if pool do
        # MLIR requires threading to be disabled before replacing its
        # internally-owned pool; setting the external pool enables it again.
        mlirContextEnableMultithreading(ctx, false)
        mlirContextSetThreadPool(ctx, pool)
      end

      if not threading, do: mlirContextEnableMultithreading(ctx, false)

      opts
      |> Keyword.get(:registry, [])
      |> List.wrap()
      |> Enum.each(&mlirContextAppendDialectRegistry(ctx, &1))

      if all_dialects, do: load_all_dialects(ctx)
      mlirContextSetAllowUnregisteredDialects(ctx, allow_unregistered)

      %{ctx | thread_pool_owner: owner, thread_pool_lease: lease}
    rescue
      exception ->
        mlirContextDestroy(ctx)
        if owner, do: MLIR.ThreadPool.checkin(owner, lease)
        reraise exception, __STACKTRACE__
    end
  end

  def destroy(%__MODULE__{} = ctx) do
    try do
      mlirContextDestroy(ctx)
    after
      MLIR.ExternalInterface.release_context(ctx)

      if ctx.thread_pool_owner do
        MLIR.ThreadPool.checkin(ctx.thread_pool_owner, ctx.thread_pool_lease)
      end
    end
  end

  defp resolve_thread_pool(_opts, false), do: {nil, nil, nil}

  defp resolve_thread_pool(opts, true) do
    case Keyword.get(opts, :thread_pool, :application) do
      nil ->
        {nil, nil, nil}

      %MLIR.LLVMThreadPool{} = pool ->
        {pool, nil, nil}

      :application ->
        checkout_if_running(MLIR.ThreadPool.default_name())

      owner when is_atom(owner) or is_pid(owner) or is_tuple(owner) ->
        checkout(owner)
    end
  end

  defp checkout_if_running(owner) do
    case GenServer.whereis(owner) do
      nil ->
        {nil, nil, nil}

      _pid ->
        checkout(owner)
    end
  end

  defp checkout(owner) do
    case MLIR.ThreadPool.checkout(owner) do
      {%MLIR.LLVMThreadPool{} = pool, lease} ->
        {pool, owner, lease}

      {:error, :closing} ->
        raise ArgumentError, "the selected MLIR thread pool is closing"
    end
  end

  def allow_unregistered_dialects(ctx, allow \\ true) do
    mlirContextSetAllowUnregisteredDialects(ctx, allow)
    ctx
  end

  def register_translations(ctx) do
    mlirRegisterAllLLVMTranslations(ctx)
    ctx
  end

  @doc """
  Check if the op name is terminator
  """
  def terminator?(%__MODULE__{} = ctx, op) do
    beaverIsOpNameTerminator(MLIR.StringRef.create(op), ctx) |> Beaver.Native.to_term()
  end

  def implements_interface?(ctx, op, interface_id) do
    mlirOperationImplementsInterfaceStatic(MLIR.StringRef.create(op), ctx, interface_id)
    |> Beaver.Native.to_term()
  end

  def infer_type?(%__MODULE__{} = ctx, op) do
    implements_interface?(ctx, MLIR.StringRef.create(op), mlirInferTypeOpInterfaceTypeID())
  end

  def infer_shaped?(%__MODULE__{} = ctx, op) do
    implements_interface?(ctx, MLIR.StringRef.create(op), mlirInferShapedTypeOpInterfaceTypeID())
  end
end
