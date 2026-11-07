alias Beaver.MLIR.StringRef
alias Beaver.Native

defmodule Beaver.MLIR.Context do
  @moduledoc """
  This module defines functions creating or destroying MLIR context.
  """
  alias Beaver.MLIR
  import MLIR.CAPI

  use Kinda.ResourceKind,
    raw_module: Beaver.MLIR.CAPI.Raw,
    codec: Native,
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
          | {:thread_safe, boolean()}
          | {:callback_executor, keyword()}
          | {:thread_pool, :application | MLIR.LLVMThreadPool.t() | GenServer.server() | nil}
          | {:registry, MLIR.DialectRegistry.t() | [MLIR.DialectRegistry.t()]}
  @spec create([context_option()]) :: __MODULE__.t()
  @doc """
  Create a MLIR context. By default it registers and loads all dialects.

  `threading: false, thread_safe: true` keeps context uniquing safe for external
  workers without creating an MLIR thread pool. Load dialects before sharing the
  context; this does not make concurrent mutation of the same IR safe.
  Beaver's callback executor also works in this mode. Use
  `Beaver.Composer.run/2` with `execution: :beam` to run Elixir passes directly.

  `callback_executor: [max_concurrency: n, idle_timeout_ms: ms]` limits admitted
  worker jobs. Jobs waiting for BEAM lend their admission slot to nested or
  queued work. Blocked threads can exceed this limit; idle workers retire.
  Dirty schedulers and MLIR's internal parallelism have independent limits.
  """
  def create(opts \\ []) do
    allow_unregistered = Keyword.get(opts, :allow_unregistered, false)
    all_dialects = Keyword.get(opts, :all_dialects, true)
    {threading, thread_safe} = threading_options(opts)
    {pool, owner, lease} = resolve_thread_pool(opts, threading)
    ctx = mlirContextCreateWithThreading(threading and is_nil(pool))

    try do
      if pool do
        # MLIR requires threading to be disabled before replacing its
        # internally-owned pool; setting the external pool enables it again.
        mlirContextEnableMultithreading(ctx, false)
        mlirContextSetThreadPool(ctx, pool)
      end

      if thread_safe and not threading, do: beaverContextEnableThreadSafeLocking(ctx)

      if executor = opts[:callback_executor], do: configure_executor(ctx, executor)

      opts |> Keyword.get(:registry, []) |> List.wrap() |> Enum.each(&mlirContextAppendDialectRegistry(ctx, &1))

      if all_dialects, do: load_all_dialects(ctx)
      mlirContextSetAllowUnregisteredDialects(ctx, allow_unregistered)

      %{ctx | thread_pool_owner: owner, thread_pool_lease: lease}
    rescue
      exception ->
        beaver_raw_executor_release(ctx.ref)
        mlirContextDestroy(ctx)
        if owner, do: MLIR.ThreadPool.checkin(owner, lease)
        reraise exception, __STACKTRACE__
    end
  end

  defp configure_executor(ctx, executor) do
    unless Keyword.keyword?(executor) and Keyword.keys(executor) -- [:max_concurrency, :idle_timeout_ms] == [],
      do: raise(ArgumentError, "invalid callback executor options")

    capacity = Keyword.get(executor, :max_concurrency, System.schedulers_online())
    idle = Keyword.get(executor, :idle_timeout_ms, 1000)

    unless is_integer(capacity) and capacity > 0 and is_integer(idle) and idle > 0,
      do: raise(ArgumentError, "callback executor capacity and idle timeout must be positive integers")

    true = beaverExecutorConfigure(ctx, capacity, idle) |> Native.to_term()
  end

  @doc "Whether the linked LLVM supports context locking without internal parallelism."
  def independent_locking_supported?, do: beaverContextIndependentLockingSupported() |> Native.to_term()

  @doc "Whether context uniquing is protected by thread-safe locks."
  def thread_safe?(%__MODULE__{} = ctx), do: beaverContextIsThreadSafeLockingEnabled(ctx) |> Native.to_term()

  def destroy(%__MODULE__{} = ctx) do
    if beaverContextHasActiveTransientScope(ctx) |> Native.to_term(),
      do: raise(ArgumentError, "cannot destroy a context with an active transient scope")

    unless beaver_raw_executor_release(ctx.ref),
      do: raise(ArgumentError, "cannot destroy a context with active callback executor work")

    try do
      # Detach action tracing sessions while the native context is still alive;
      # the native session destructor deregisters its action handler on the
      # context.
      MLIR.ActionTracing.release_context(ctx)
      mlirContextDestroy(ctx)
    after
      MLIR.Trait.release_context(ctx)
      MLIR.ExternalInterface.release_context(ctx)

      if ctx.thread_pool_owner,
        do: MLIR.ThreadPool.checkin(ctx.thread_pool_owner, ctx.thread_pool_lease)
    end
  end

  @doc "Current callback executor capacity, queue and worker counts for this context."
  defdelegate executor_stats(ctx), to: Beaver.Execution

  @doc "Returns whether the linked LLVM supports resettable transient context scopes."
  @spec transient_scope_supported?() :: boolean()
  def transient_scope_supported?,
    do: beaverContextTransientScopeSupported() |> Native.to_term()

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
    unless transient_scope_supported?(),
      do: raise(ArgumentError, "linked LLVM does not support transient context scopes")

    unless beaverContextBeginTransientScope(ctx) |> Native.to_term(),
      do: raise(ArgumentError, "context already has an active transient scope")

    try do
      fun.(ctx)
    after
      unless beaverContextEndTransientScope(ctx) |> Native.to_term(),
        do: raise("transient context scope ended unexpectedly")
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

  def ensure_same!({:ok, entity}, %__MODULE__{} = expected), do: {:ok, ensure_same!(entity, expected)}

  def ensure_same!(entities, %__MODULE__{} = expected) when is_list(entities),
    do: Enum.map(entities, &ensure_same!(&1, expected))

  def ensure_same!(entity, %__MODULE__{}), do: entity

  defp threading_options(opts) do
    threading = Keyword.get(opts, :threading, true)
    thread_safe = Keyword.get(opts, :thread_safe, threading)

    unless is_boolean(threading) and is_boolean(thread_safe),
      do: raise(ArgumentError, ":threading and :thread_safe must be booleans")

    if threading and not thread_safe,
      do: raise(ArgumentError, "threading: true requires thread_safe: true")

    if thread_safe and not threading and not independent_locking_supported?(),
      do: raise(ArgumentError, "linked LLVM does not support independent context locking")

    {threading, thread_safe}
  end

  defp resolve_thread_pool(opts, false) do
    if Keyword.has_key?(opts, :thread_pool),
      do: raise(ArgumentError, ":thread_pool cannot be used with threading: false")

    {nil, nil, nil}
  end

  defp resolve_thread_pool(opts, true) do
    case Keyword.get(opts, :thread_pool, :application) do
      nil -> {nil, nil, nil}
      %MLIR.LLVMThreadPool{} = pool -> {pool, nil, nil}
      :application -> checkout_if_running(MLIR.ThreadPool.default_name())
      owner when is_atom(owner) or is_pid(owner) or is_tuple(owner) -> checkout(owner)
    end
  end

  defp checkout_if_running(owner) do
    case GenServer.whereis(owner) do
      nil -> {nil, nil, nil}
      _pid -> checkout(owner)
    end
  end

  defp checkout(owner) do
    case MLIR.ThreadPool.checkout(owner) do
      {%MLIR.LLVMThreadPool{} = pool, lease} -> {pool, owner, lease}
      {:error, :closing} -> raise ArgumentError, "the selected MLIR thread pool is closing"
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
  def terminator?(%__MODULE__{} = ctx, op), do: MLIR.Trait.has?(ctx, op, :terminator)

  def implements_interface?(ctx, op, interface_id),
    do: mlirOperationImplementsInterfaceStatic(StringRef.create(op), ctx, interface_id) |> Native.to_term()

  def infer_type?(%__MODULE__{} = ctx, op),
    do: implements_interface?(ctx, StringRef.create(op), mlirInferTypeOpInterfaceTypeID())

  def infer_shaped?(%__MODULE__{} = ctx, op),
    do: implements_interface?(ctx, StringRef.create(op), mlirInferShapedTypeOpInterfaceTypeID())
end
