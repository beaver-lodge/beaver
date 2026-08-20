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
    if beaverContextHasActiveTransientScope(ctx) |> Beaver.Native.to_term() do
      raise ArgumentError, "cannot destroy a context with an active transient scope"
    end

    try do
      # Detach action tracing sessions while the native context is still alive;
      # the native session destructor deregisters its action handler on the
      # context.
      MLIR.ActionTracing.release_context(ctx)
      mlirContextDestroy(ctx)
    after
      MLIR.Trait.release_context(ctx)
      MLIR.ExternalInterface.release_context(ctx)

      if ctx.thread_pool_owner do
        MLIR.ThreadPool.checkin(ctx.thread_pool_owner, ctx.thread_pool_lease)
      end
    end
  end

  @doc "Returns whether the linked LLVM supports resettable transient context scopes."
  @spec transient_scope_supported?() :: boolean()
  def transient_scope_supported? do
    beaverContextTransientScopeSupported()
    |> Beaver.Native.to_term()
  end

  @doc """
  Runs `fun` inside a resettable transient allocation scope.

  Dialects and external interfaces must be registered before entering the
  scope. All IR that refers to types or attributes created inside the scope
  must be destroyed before `fun` returns. The same context cannot host
  concurrent or nested transient scopes. Types, attributes, and other
  non-owning handles created inside the scope become stale when it ends and
  must not be dereferenced; allowing their wrappers to be garbage-collected
  afterward is safe.
  """
  @spec with_transient_scope(t(), (t() -> result)) :: result when result: term()
  def with_transient_scope(%__MODULE__{} = ctx, fun) when is_function(fun, 1) do
    unless transient_scope_supported?() do
      raise ArgumentError, "linked LLVM does not support transient context scopes"
    end

    unless beaverContextBeginTransientScope(ctx) |> Beaver.Native.to_term() do
      raise ArgumentError, "context already has an active transient scope"
    end

    try do
      fun.(ctx)
    after
      unless beaverContextEndTransientScope(ctx) |> Beaver.Native.to_term() do
        raise "transient context scope ended unexpectedly"
      end
    end
  end

  @context_owned_modules [
    MLIR.AffineExpr,
    MLIR.AffineMap,
    MLIR.Attribute,
    MLIR.Dialect,
    MLIR.Identifier,
    MLIR.IntegerSet,
    MLIR.Location,
    MLIR.Module,
    MLIR.Operation,
    MLIR.Type,
    MLIR.Value
  ]

  @doc "Returns whether two handles refer to the same MLIR context."
  @spec same?(t(), t()) :: boolean()
  def same?(%__MODULE__{} = left, %__MODULE__{} = right), do: MLIR.equal?(left, right)

  @doc """
  Verifies that a context-owned MLIR entity belongs to `expected`.

  Context-free values pass through unchanged, which makes this suitable for
  validating the output of contextual builders at their common boundary.
  """
  @spec ensure_same!(term(), t()) :: term()
  def ensure_same!(%module{} = entity, %__MODULE__{} = expected)
      when module in @context_owned_modules do
    if not MLIR.null?(entity) and not same?(MLIR.context(entity), expected) do
      kind = module |> Module.split() |> List.last() |> Macro.underscore()
      raise ArgumentError, "#{kind} belongs to a different MLIR context"
    end

    entity
  end

  def ensure_same!({:ok, entity}, %__MODULE__{} = expected),
    do: {:ok, ensure_same!(entity, expected)}

  def ensure_same!(entities, %__MODULE__{} = expected) when is_list(entities),
    do: Enum.map(entities, &ensure_same!(&1, expected))

  def ensure_same!(entity, %__MODULE__{}), do: entity

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
    MLIR.Trait.has?(ctx, op, :terminator)
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
