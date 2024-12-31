defmodule Beaver.Native do
  @moduledoc """
  This module provide higher level interface to primitive data types and MemRefDescriptor C struct in [MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/) dialect.
  It aims to manage memory with BEAM and help a Erlang/Elixir function play well with function generated by MLIR and other NIFs.
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
        Enum.map(data, &Kinda.unwrap_ref/1)
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

  defp postprocess_diagnostics({severity_i, loc_ref, note, nested}) do
    {Beaver.MLIR.Diagnostic.severity(severity_i), %Beaver.MLIR.Location{ref: loc_ref},
     to_string(note), Enum.map(nested, &postprocess_diagnostics/1)}
  end

  def check!(ret) do
    case ret do
      {:kind, mod, ref} when is_atom(mod) and is_reference(ref) ->
        try do
          struct!(mod, %{ref: ref})
        rescue
          UndefinedFunctionError -> ref
        end

      {{:kind, mod, ref}, diagnostics} ->
        {try do
           struct!(mod, %{ref: ref})
         rescue
           UndefinedFunctionError -> ref
         end, Enum.map(diagnostics, &postprocess_diagnostics/1)}

      ret ->
        ret
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

  def dump(%kind{ref: ref}) do
    Beaver.Native.forward(kind, "dump", [ref])
  end

  def apply_dirty(fun, args, dirty_flag) do
    f =
      case dirty_flag do
        :io_bound ->
          :"#{fun}_dirty_io"

        :cpu_bound ->
          :"#{fun}_dirty_cpu"

        nil ->
          fun
      end

    apply(CAPI, f, args)
  end
end
