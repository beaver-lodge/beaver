defmodule Beaver.Native do
  @moduledoc """
  This module provide higher level interface to primitive data types and MemRefDescriptor C struct in [MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/) dialect.
  It aims to manage memory with BEAM utility like ets and help a Erlang/Elixir function play well with function generated by MLIR and other NIFs.
  """

  alias Beaver.MLIR.CAPI

  def ptr(%mod{ref: ref}) do
    %__MODULE__.Ptr{
      ref: apply(CAPI, Module.concat(mod, :ptr), [ref]) |> check!(),
      element_kind: mod
    }
  end

  def opaque_ptr(%mod{ref: ref}) do
    maker = Module.concat([mod, :opaque_ptr])

    %__MODULE__.OpaquePtr{
      ref: apply(CAPI, maker, [ref]) |> check!()
    }
  end

  def opaque_ptr(%__MODULE__.Array{} = array) do
    array
    |> __MODULE__.Array.as_opaque()
  end

  def array(data, module, opts \\ [mut: false])

  def array(data, module, opts) when is_binary(data) do
    mut = Keyword.get(opts, :mut) || false
    func = if mut, do: "mut_array", else: "array"

    ref =
      apply(CAPI, Module.concat([module, func]), [
        data
      ])
      |> check!()

    %__MODULE__.Array{ref: ref, element_kind: module}
  end

  def array(data, module, opts) when is_list(data) do
    mut = Keyword.get(opts, :mut) || false
    func = if mut, do: "mut_array", else: "array"

    ref =
      apply(CAPI, Module.concat([module, func]), [
        Enum.map(data, &Fizz.unwrap_ref/1)
      ])
      |> check!()

    %__MODULE__.Array{ref: ref, element_kind: module}
  end

  def to_term(%__MODULE__.Ptr{ref: ref, element_kind: __MODULE__.OpaquePtr}) do
    forward(__MODULE__.OpaquePtr, :primitive, [ref])
  end

  def to_term(%mod{ref: ref}) do
    forward(mod, :primitive, [ref])
  end

  def bag(%{bag: bag} = v, list) when is_list(list) do
    %{v | bag: MapSet.union(MapSet.new(list), bag)}
  end

  def bag(%{bag: bag} = v, item) do
    %{v | bag: MapSet.put(bag, item)}
  end

  def c_string(value) when is_binary(value) do
    %__MODULE__.C.String{ref: check!(CAPI.beaver_raw_get_resource_c_string(value))}
  end

  def check!(ref) do
    case ref do
      {:error, e} ->
        raise e

      ref ->
        ref
    end
  end

  def forward(
        element_kind,
        kind_func_name,
        args
      ) do
    apply(CAPI, Module.concat(element_kind, kind_func_name), args)
    |> check!()
  end

  def dump(%{kind: kind, ref: ref} = v) do
    Beaver.Native.forward(kind, "dump", [ref])
    v
  end
end
