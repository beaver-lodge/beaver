defmodule Beaver.MLIR.Global.Context do
  alias Beaver.MLIR
  use Agent

  def start_link([]) do
    # TODO: read opts from app config
    Agent.start_link(fn -> MLIR.Context.create(allow_unregistered: true) end, name: __MODULE__)
  end

  def get do
    Agent.get(__MODULE__, & &1)
  end
end

defmodule Beaver.MLIR.Managed.Context do
  @moduledoc """
  Getting and setting managed MLIR context
  """
  def get() do
    case Process.get(__MODULE__) do
      nil ->
        global = Beaver.MLIR.Global.Context.get()
        set(global)
        global

      managed ->
        managed
    end
  end

  def set(ctx) do
    Process.put(__MODULE__, ctx)
    ctx
  end

  def from_opts(opts) when is_list(opts) do
    ctx = opts[:ctx]

    if ctx do
      ctx
    else
      get()
    end
  end
end

defmodule Beaver.MLIR.Managed.Location do
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Getting and setting managed MLIR location
  """
  def get() do
    case Process.get(__MODULE__) do
      nil ->
        ctx = Beaver.MLIR.Managed.Context.get()
        location = CAPI.mlirLocationUnknownGet(ctx)
        set(location)

      managed ->
        managed
    end
  end

  def set(location) do
    Process.put(__MODULE__, location)
    location
  end

  def from_opts(opts) when is_list(opts) do
    location = opts[:loc]

    if location do
      location
    else
      get()
    end
  end
end

defmodule Beaver.MLIR.Managed.Region do
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Getting and setting managed MLIR region
  """
  def get() do
    case Process.get(__MODULE__) do
      nil ->
        raise "region not set, create and set it with Beaver.MLIR.Managed.Region.set/1"

      managed ->
        managed
    end
  end

  @doc """
  Set a region as the contextual region. Usaully it's recommended to call unset/0 to make sure the region is not used by other op.
  """
  def set(region) do
    Process.put(__MODULE__, region)
    region
  end

  def unset() do
    Process.delete(__MODULE__)
  end

  def from_opts(opts) when is_list(opts) do
    region = opts[:region]

    if region do
      region
    else
      get()
    end
  end
end

defmodule Beaver.MLIR.Managed.InsertionPoint do
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Getting and setting insertion point. In high level APIs in Beaver.MLIR, an insertion point in is usually an anonymous function.
  """
  def get() do
    case Process.get(__MODULE__) do
      nil ->
        raise "insertion point not set, create and call Beaver.MLIR.Managed.InsertionPoint.push/1"

      [] ->
        raise "insertion point empty, create and call Beaver.MLIR.Managed.InsertionPoint.push/1"

      managed ->
        [head | _tail] = managed
        head
    end
  end

  @doc """
  Push an insert_point. Usaully it's recommended to call pop/0 and empty?/0 to make sure the insertion point is not used by other op.
  """
  def push(insert_point) when is_function(insert_point) do
    old =
      case Process.delete(__MODULE__) do
        nil ->
          []

        managed ->
          managed
      end

    new = [insert_point | old]
    Process.put(__MODULE__, new)
    new
  end

  def empty?() do
    managed = Process.get(__MODULE__)
    managed == [] or is_nil(managed)
  end

  def pop() do
    managed = Process.get(__MODULE__)

    case managed do
      [] ->
        raise "insertion point empty"

      nil ->
        raise "insertion point is nil"

      managed ->
        [head | tail] = managed
        Process.put(__MODULE__, tail)
        head
    end
  end

  def from_opts(opts) when is_list(opts) do
    insert_point = opts[:insert_point]

    if insert_point do
      insert_point
    else
      get()
    end
  end
end
