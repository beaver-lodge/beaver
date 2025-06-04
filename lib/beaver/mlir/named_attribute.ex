defmodule Beaver.MLIR.NamedAttribute do
  @moduledoc """
  This module defines a wrapper struct of NamedAttribute in MLIR
  """
  alias Beaver.MLIR
  import MLIR.CAPI
  use Kinda.ResourceKind, forward_module: Beaver.Native

  @doc """
  create named attribute.

  will try to use context extracted from `name` and `attr` if `opts` is empty
  """
  def get(name, attr, opts \\ [])

  def get(%MLIR.Identifier{} = name, attr, []) do
    get(name, attr, ctx: MLIR.context(name))
  end

  def get(name, %MLIR.Attribute{} = attr, []) do
    get(name, attr, ctx: MLIR.context(attr))
  end

  def get(name, attr, opts) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        mlirNamedAttributeGet(
          case name do
            %MLIR.Identifier{} -> name
            s -> MLIR.Identifier.get(s, ctx: ctx)
          end,
          Beaver.Deferred.create(attr, ctx)
        )
      end
    )
  end

  defdelegate name(named_attribute), to: MLIR.CAPI, as: :beaverNamedAttributeGetName
  defdelegate attribute(named_attribute), to: MLIR.CAPI, as: :beaverNamedAttributeGetAttribute
end
