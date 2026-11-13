defmodule Beaver.MLIR.Region do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  @moduledoc "This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}."
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  def under(region, f) when is_function(f, 0) do
    f.()
    region
  end

  @doc "Creates a new empty detached `MLIR.Region`."
  @spec create() :: t()
  defdelegate create(), to: CAPI, as: :mlirRegionCreate
  @doc "Destroys a detached `MLIR.Region`."
  @spec destroy(t()) :: :ok
  defdelegate destroy(region), to: CAPI, as: :mlirRegionDestroy
  defdelegate append(region, block), to: CAPI, as: :mlirRegionAppendOwnedBlock
  defdelegate insert(region, index, block), to: CAPI, as: :mlirRegionInsertOwnedBlock
  @doc "Returns the first block in `region`, or `nil` if the region is empty."
  @spec first_block(t()) :: MLIR.Block.t() | nil
  def first_block(%__MODULE__{} = region) do
    block = CAPI.mlirRegionGetFirstBlock(region)
    if MLIR.null?(block), do: nil, else: block
  end

  @doc "Returns an enumerable of all blocks in `region`."
  @spec blocks(t()) :: Enumerable.t()
  def blocks(%__MODULE__{} = region), do: Beaver.Walker.blocks(region)
  @doc "Creates a block with optional argument specifications, appends it to `region`, and returns the block."
  @spec append_block(t(), [MLIR.Block.arg()], keyword()) :: MLIR.Block.t()
  def append_block(%__MODULE__{} = region, arg_specs \\ [], opts \\ [])
      when is_list(arg_specs) and is_list(opts) do
    block = MLIR.Block.create()
    append(region, block)
    if arg_specs != [], do: MLIR.Block.add_args!(block, arg_specs, opts)
    block
  end
end
