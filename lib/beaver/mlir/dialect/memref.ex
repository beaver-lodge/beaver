defmodule Beaver.MLIR.Dialect.MemRef do
  use Beaver
  alias Beaver.MLIR.Attribute
  alias Beaver.MLIR.Dialect

  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """

  use Beaver.MLIR.Dialect,
    dialect: "memref",
    ops: Dialect.Registry.ops("memref")

  @global "memref.global"
  @doc """
  Create a global.

  ## Special arguments
  - `global(binary(), {[:i | :f], [8 | 16 | 32 | 64 | 128]})`: To to serialize a binary to MLIR. By default, it will be a `memref<[byte size]*i8>`.
  """
  def global(%Beaver.SSA{arguments: [txt | arguments]} = ssa)
      when is_binary(txt) do
    {t, width} = {:i, 8}
    value = Attribute.dense_elements(txt, {t, width})
    sym_name = :erlang.md5(txt) |> Base.encode16(case: :lower)

    arguments =
      arguments
      |> Keyword.put_new(:sym_name, Attribute.string(sym_name))
      |> Keyword.put_new(:initial_value, value)
      |> Keyword.put_new(:type, ~t{memref<#{byte_size(txt)}x#{t}#{width}>})

    %Beaver.SSA{ssa | op: @global, arguments: arguments, results: []} |> Beaver.SSA.eval()
  end

  def global(%Beaver.SSA{} = ssa) do
    Beaver.SSA.eval(%Beaver.SSA{ssa | op: @global})
  end
end
