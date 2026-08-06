defmodule Beaver.MLIR.ThreadPool do
  @moduledoc """
  Owns an LLVM thread pool that can be shared by multiple MLIR contexts.

  The application starts one pool under `default_name/0`. `MLIR.Context.create/1`
  checks out a lease from it by default and returns the lease when the context is
  destroyed. Calling `close/1` while contexts are attached defers native pool
  destruction until the final lease is returned.

  If an owner is forcibly terminated while leases remain, the native pool is
  deliberately retained so attached MLIR contexts never hold a dangling pool
  pointer. Use `close/1` for deterministic reclamation.

  An explicit owner can be selected with `thread_pool: owner`, or a raw
  `Beaver.MLIR.LLVMThreadPool` can be supplied when its lifetime is managed by
  the caller.
  """

  use GenServer

  alias Beaver.MLIR

  @default_name __MODULE__

  @spec default_name() :: module()
  def default_name, do: @default_name

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    {name, opts} = Keyword.pop(opts, :name)
    GenServer.start_link(__MODULE__, opts, if(name, do: [name: name], else: []))
  end

  @spec child_spec(keyword()) :: Supervisor.child_spec()
  def child_spec(opts) do
    %{
      id: Keyword.get(opts, :name, __MODULE__),
      start: {__MODULE__, :start_link, [opts]},
      restart: :transient,
      type: :worker
    }
  end

  @spec checkout(GenServer.server()) ::
          {MLIR.LLVMThreadPool.t(), reference()} | {:error, :closing}
  def checkout(owner \\ @default_name), do: GenServer.call(owner, :checkout)

  @spec checkin(GenServer.server(), reference()) :: :ok
  def checkin(owner \\ @default_name, lease) do
    GenServer.cast(owner, {:checkin, lease})
  catch
    :exit, _ -> :ok
  end

  @spec pool(GenServer.server()) :: MLIR.LLVMThreadPool.t()
  def pool(owner \\ @default_name), do: GenServer.call(owner, :pool)

  @doc "Create a caller-owned native pool for use with `MLIR.Context.create/1`."
  @spec create() :: MLIR.LLVMThreadPool.t()
  def create, do: MLIR.CAPI.mlirLlvmThreadPoolCreate()

  @doc "Destroy a caller-owned native pool after all attached contexts are gone."
  @spec destroy(MLIR.LLVMThreadPool.t()) :: :ok
  defdelegate destroy(pool), to: MLIR.CAPI, as: :mlirLlvmThreadPoolDestroy

  @spec max_concurrency(GenServer.server() | MLIR.LLVMThreadPool.t()) :: non_neg_integer()
  def max_concurrency(owner \\ @default_name)

  def max_concurrency(%MLIR.LLVMThreadPool{} = pool) do
    MLIR.CAPI.mlirLlvmThreadPoolGetMaxConcurrency(pool) |> Beaver.Native.to_term()
  end

  def max_concurrency(owner), do: owner |> pool() |> max_concurrency()

  @doc """
  Closes the owner. Returns `:deferred` if attached contexts still hold leases.
  """
  @spec close(GenServer.server()) :: :ok | :deferred
  def close(owner \\ @default_name), do: GenServer.call(owner, :close)

  @impl GenServer
  def init(opts) do
    concurrency = Keyword.get(opts, :concurrency, System.schedulers_online())

    unless is_integer(concurrency) and concurrency > 0 do
      raise ArgumentError, ":concurrency must be a positive integer"
    end

    pool = MLIR.CAPI.beaverLlvmThreadPoolCreateElastic(concurrency)
    {:ok, %{pool: pool, leases: MapSet.new(), closing?: false}}
  end

  @impl GenServer
  def handle_call(:pool, _from, state), do: {:reply, state.pool, state}

  def handle_call(:checkout, _from, %{closing?: true} = state) do
    {:reply, {:error, :closing}, state}
  end

  def handle_call(:checkout, _from, state) do
    lease = make_ref()
    {:reply, {state.pool, lease}, %{state | leases: MapSet.put(state.leases, lease)}}
  end

  def handle_call(:close, _from, %{leases: leases} = state) do
    if MapSet.size(leases) == 0 do
      {:stop, :normal, :ok, state}
    else
      {:reply, :deferred, %{state | closing?: true}}
    end
  end

  @impl GenServer
  def handle_cast({:checkin, lease}, state) do
    state = %{state | leases: MapSet.delete(state.leases, lease)}

    if state.closing? and MapSet.size(state.leases) == 0 do
      {:stop, :normal, state}
    else
      {:noreply, state}
    end
  end

  @impl GenServer
  def terminate(_reason, state) do
    if MapSet.size(state.leases) == 0 do
      destroy(state.pool)
    end

    :ok
  end
end
