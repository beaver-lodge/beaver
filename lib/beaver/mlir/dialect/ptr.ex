defmodule Beaver.MLIR.Dialect.Ptr do
  @moduledoc """
  Operations and construction helpers for MLIR's upstream Ptr dialect.

  Ptr values model native pointers in IR. They are distinct from
  `Beaver.Native.OpaquePtr` and other BEAM resources, which own or reference
  host-side native data outside MLIR.

  Use `type/1` with the generic memory space at high-level ABI boundaries such
  as ENIF calls. Use an explicit LLVM address space when Ptr operations remain
  in an `llvm.func` until LLVM IR translation:

      Ptr.type()
      Ptr.type(memory_space: {:llvm, 3})

  `null/1` and `address/2` return typed attributes suitable for
  `ptr.constant`.
  """

  alias Beaver.Deferred
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Dialect, Type}

  use Dialect, dialect: "ptr", ops: Dialect.Registry.ops("ptr")

  @type memory_space ::
          :generic
          | non_neg_integer()
          | {:llvm, non_neg_integer()}
          | Deferred.attribute()

  @doc "Return the upstream Ptr generic-memory-space attribute."
  @spec generic_space(keyword()) :: Deferred.attribute()
  def generic_space(opts \\ []) do
    Attribute.get("#ptr.generic_space", opts)
  end

  @doc "Return an LLVM address-space attribute for a non-negative address-space number."
  @spec llvm_address_space(non_neg_integer(), keyword()) :: Deferred.attribute()
  def llvm_address_space(address_space, opts \\ [])

  def llvm_address_space(address_space, opts)
      when is_integer(address_space) and address_space >= 0 do
    Attribute.get("#llvm.address_space<#{address_space}>", opts)
  end

  def llvm_address_space(address_space, _opts) do
    raise ArgumentError,
          "address space must be a non-negative integer, got: #{inspect(address_space)}"
  end

  @doc """
  Return a `!ptr.ptr` type.

  The default is `#ptr.generic_space`. Pass either an address-space number or
  `{:llvm, address_space}` to make the LLVM address space explicit.
  """
  @spec type(keyword()) :: Deferred.type()
  def type(opts \\ []) do
    memory_space = Keyword.get(opts, :memory_space, :generic)

    Deferred.from_opts(opts, fn ctx ->
      memory_space = memory_space |> materialize_memory_space(ctx) |> MLIR.to_string()
      Type.get("!ptr.ptr<#{memory_space}>", ctx: ctx)
    end)
  end

  @doc "Return a typed `#ptr.null` attribute for `ptr.constant`."
  @spec null(keyword()) :: Deferred.attribute()
  def null(opts \\ []) do
    constant_attribute("#ptr.null", opts)
  end

  @doc "Return a typed `#ptr.address` attribute for `ptr.constant`."
  @spec address(non_neg_integer(), keyword()) :: Deferred.attribute()
  def address(value, opts \\ [])

  def address(value, opts) when is_integer(value) and value >= 0 do
    constant_attribute("#ptr.address<#{value}>", opts)
  end

  def address(value, _opts) do
    raise ArgumentError, "pointer address must be a non-negative integer, got: #{inspect(value)}"
  end

  defp constant_attribute(source, opts) do
    pointer_type =
      Keyword.get_lazy(opts, :type, fn ->
        type(memory_space: Keyword.get(opts, :memory_space, :generic))
      end)

    Deferred.from_opts(opts, fn ctx ->
      pointer_type = Deferred.resolve(pointer_type, ctx)
      Attribute.get("#{source} : #{MLIR.to_string(pointer_type)}", ctx: ctx)
    end)
  end

  defp materialize_memory_space(:generic, ctx), do: generic_space(ctx: ctx)

  defp materialize_memory_space({:llvm, address_space}, ctx),
    do: llvm_address_space(address_space, ctx: ctx)

  defp materialize_memory_space(address_space, ctx) when is_integer(address_space),
    do: llvm_address_space(address_space, ctx: ctx)

  defp materialize_memory_space(memory_space, ctx), do: Deferred.resolve(memory_space, ctx)
end
