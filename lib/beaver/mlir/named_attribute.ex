alias Beaver.MLIR.CAPI
alias Beaver.Deferred

defmodule Beaver.MLIR.NamedAttribute do
  @moduledoc """
  This module defines a wrapper struct of NamedAttribute in MLIR
  """
  alias Beaver.MLIR
  import CAPI
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Beaver.Native

  @doc """
  create named attribute.

  will try to use context extracted from `name` and `attr` if `opts` is empty
  """
  def get(name, attr, opts \\ [])

  def get(%MLIR.Identifier{} = name, attr, []), do: get(name, attr, ctx: MLIR.context(name))

  def get(name, %MLIR.Attribute{} = attr, []), do: get(name, attr, ctx: MLIR.context(attr))

  def get(name, attr, opts) do
    Deferred.from_opts(
      opts,
      fn ctx ->
        identifier =
          case name do
            %MLIR.Identifier{} -> name
            s -> MLIR.Identifier.get(s, ctx: ctx)
          end

        mlirNamedAttributeGet(Deferred.resolve(identifier, ctx), Deferred.resolve(attr, ctx))
      end
    )
  end

  defdelegate name(named_attribute), to: CAPI, as: :beaverNamedAttributeGetName
  defdelegate attribute(named_attribute), to: CAPI, as: :beaverNamedAttributeGetAttribute
end
