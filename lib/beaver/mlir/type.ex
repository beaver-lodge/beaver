alias Beaver.MLIR.{CAPI, StringRef}
alias Beaver.{Deferred, Native}

defmodule Beaver.MLIR.Type do
  @moduledoc """
  This module provides functions to work with MLIR's type system, allowing creation of MLIR type.

  ## Type Categories

  ### Basic Types
  - `integer/1` (`i32/1`, `i64/1`, etc.)
  - `float/1` (`f32/1`, `f64/1`, etc.)
  - `index/1`
  - `none/1`

  ### Composite Types
  - Tensors (`ranked_tensor/2` and `unranked_tensor/1`)
  - `vector/2`
  - `memref/3`
  - `function/3`
  """
  alias Beaver.MLIR
  import CAPI

  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native

  def get(string, opts \\ [])

  def get(string, opts) when is_binary(string) do
    Deferred.from_opts(opts, fn ctx ->
      t = mlirTypeParseGet(ctx, StringRef.create(string))

      if MLIR.null?(t), do: raise("fail to parse type: #{string}")

      t
    end)
  end

  def function(inputs, results, opts \\ []) do
    inputs = List.wrap(inputs)
    results = List.wrap(results)
    num_inputs = length(inputs)
    num_results = length(results)

    Deferred.from_opts(opts, fn ctx ->
      inputs =
        inputs |> Enum.map(&Deferred.resolve(&1, ctx)) |> Native.array(__MODULE__)

      results = results |> Enum.map(&Deferred.resolve(&1, ctx)) |> Native.array(__MODULE__)

      mlirFunctionTypeGet(ctx, num_inputs, inputs, num_results, results)
    end)
  end

  defp checked_composite_type(ctx, getter, args, opts) do
    loc = (opts[:loc] || MLIR.Location.unknown(ctx: ctx)) |> Deferred.resolve(ctx)
    {t, diagnostics} = apply(getter, [ctx, loc | args])

    if MLIR.null?(t) do
      {:error,
       for {_, loc, d, _} <- diagnostics, reduce: "" do
         "" -> "#{to_string(loc)}: #{d}"
         acc -> "#{acc}\n#{to_string(loc)}: #{d}"
       end}
    else
      {:ok, t}
    end
  end

  defp bang_composite_type(cb, args) do
    case apply(cb, args) do
      %Deferred{} -> raise ArgumentError, "calling a bang function to compose a type must be eager"
      {:ok, t} -> t
      {:error, msg} -> raise ArgumentError, msg
    end
  end

  # The five checked composites share deferred resolution, context checks and
  # bang semantics; only their C API argument layout differs.
  for {name, ranked?} <- [
        ranked_tensor: true,
        memref: true,
        vector: true,
        unranked_tensor: false,
        unranked_memref: false
      ] do
    shape = Macro.var(:shape, nil)
    prefix = if ranked?, do: [shape], else: []
    shape_arg = if ranked?, do: shape, else: nil

    def unquote(name)(unquote_splicing(prefix), element, opts \\ []) do
      composite(unquote(name), unquote(shape_arg), element, opts)
    end

    def unquote(:"#{name}!")(unquote_splicing(prefix), element, opts \\ []) do
      bang_composite_type(&unquote(name)(unquote_splicing(prefix), &1, &2), [element, opts])
    end
  end

  def complex(%Deferred{} = element), do: Deferred.defer(&complex(Deferred.resolve(element, &1)))
  def complex(%__MODULE__{} = element), do: mlirComplexTypeGet(element)

  defp composite(name, shape, %Deferred{} = element, opts) do
    Deferred.from_opts(opts, fn ctx ->
      composite(name, shape, Deferred.resolve(element, ctx), Keyword.put(opts, :ctx, ctx))
    end)
  end

  defp composite(name, shape, %__MODULE__{} = element, opts) do
    ctx = composite_context(element, opts)
    attr = fn key -> Deferred.resolve(opts[key] || mlirAttributeGetNull(), ctx) end

    dimensions = fn ->
      dims = if name == :vector, do: shape, else: Enum.map(shape, &MLIR.ShapedType.to_dynamic_magic_number(&1, :size))
      [length(shape), Native.array(dims, Native.I64)]
    end

    {getter, args} =
      case name do
        :ranked_tensor ->
          {&mlirRankedTensorTypeGetCheckedWithDiagnostics/6, dimensions.() ++ [element, attr.(:encoding)]}

        :memref ->
          {&mlirMemRefTypeGetCheckedWithDiagnostics/7,
           [element] ++ dimensions.() ++ [attr.(:layout), attr.(:memory_space)]}

        :vector ->
          {&mlirVectorTypeGetCheckedWithDiagnostics/5, dimensions.() ++ [element]}

        :unranked_tensor ->
          {&mlirUnrankedTensorTypeGetCheckedWithDiagnostics/3, [element]}

        :unranked_memref ->
          {&mlirUnrankedMemRefTypeGetCheckedWithDiagnostics/4, [element, attr.(:memory_space)]}
      end

    checked_composite_type(ctx, getter, args, opts)
  end

  defp composite_context(%__MODULE__{} = element_type, opts) do
    case Deferred.context(opts) do
      nil -> MLIR.context(element_type)
      %MLIR.Context{} = ctx -> element_type |> Deferred.resolve(ctx) |> MLIR.context()
    end
  end

  @doc """
  Get a tuple type.

  ## Examples
      iex> ctx = MLIR.Context.create()
      iex> MLIR.Type.tuple([MLIR.Type.i32, MLIR.Type.i32], ctx: ctx) |> MLIR.to_string()
      "tuple<i32, i32>"
      iex> MLIR.Context.destroy(ctx)
  """
  def tuple(elements, opts \\ []) when is_list(elements) do
    Deferred.from_opts(opts, fn ctx ->
      num_elements = length(elements)

      elements = elements |> Enum.map(&Deferred.resolve(&1, ctx)) |> Native.array(__MODULE__)

      mlirTupleTypeGet(ctx, num_elements, elements)
    end)
  end

  def float(bitwidth, opts \\ []) when is_integer(bitwidth),
    do: apply(__MODULE__, String.to_atom("f#{bitwidth}"), [opts])

  defdelegate f(bitwidth, opts \\ []), to: __MODULE__, as: :float

  def opaque(dialect_namespace, type_data, opts \\ []) do
    Deferred.from_opts(opts, fn ctx ->
      mlirOpaqueTypeGet(ctx, StringRef.create(dialect_namespace), StringRef.create(type_data))
    end)
  end

  def integer(bitwidth, opts \\ [signed: false]) do
    signed = Keyword.get(opts, :signed)

    Deferred.from_opts(
      opts,
      fn ctx ->
        case signed do
          nil -> mlirIntegerTypeGet(ctx, bitwidth)
          true -> mlirIntegerTypeSignedGet(ctx, bitwidth)
          false -> mlirIntegerTypeUnsignedGet(ctx, bitwidth)
        end
      end
    )
  end

  defdelegate i(bitwidth, opts \\ []), to: __MODULE__, as: :integer

  for bitwidth <- [1, 8, 16, 32, 64, 128], sign <- ~w{i si ui} do
    i_name = "#{sign}#{bitwidth}" |> String.to_atom()

    signed =
      case sign do
        "i" -> nil
        "si" -> true
        "ui" -> false
      end

    def unquote(i_name)(opts \\ []) do
      opts = Keyword.put_new(opts, :signed, unquote(signed))
      apply(__MODULE__, :i, [unquote(bitwidth), opts])
    end
  end

  for {f, "mlirTypeIsA" <> type_name, 1} <-
        CAPI.__info__(:functions) |> Enum.map(fn {f, a} -> {f, Atom.to_string(f), a} end) do
    type_name =
      type_name |> String.replace(~r"Type$", "")

    @doc """
    calls `Beaver.MLIR.CAPI.#{f}/1` to check if it is #{type_name} type.
    """
    helper_name =
      type_name |> Macro.underscore() |> String.replace("mem_ref", "memref")

    def unquote(:"#{helper_name}?")(%__MODULE__{} = t),
      do: unquote(f)(t) |> Native.to_term()
  end

  for sign_type <- ~w{signless signed unsigned} do
    f = :"mlirIntegerTypeIs#{Macro.camelize(sign_type)}"

    @doc """
    calls `Beaver.MLIR.CAPI.#{f}/1` to check if it is a #{sign_type} integer.
    """

    def unquote(:"#{sign_type}?")(%__MODULE__{} = type),
      do: unquote(f)(type) |> Native.to_term()
  end

  for {f, "mlir" <> type_name, 1} <-
        CAPI.__info__(:functions)
        |> Enum.map(fn {f, a} -> {f, Atom.to_string(f), a} end)
        |> Enum.filter(fn {_, type_name, _} ->
          String.ends_with?(type_name, "TypeGet") and not String.contains?(type_name, "Complex") and
            not String.contains?(type_name, "UnrankedTensor")
        end) do
    type_name =
      type_name |> String.trim_trailing("TypeGet")

    @doc """
    calls `Beaver.MLIR.CAPI.#{f}/1` to get #{type_name} type
    """
    helper_name =
      type_name |> Macro.underscore() |> String.replace("mem_ref", "memref")

    def unquote(:"#{helper_name}")(opts \\ []),
      do: Deferred.from_opts(opts, &unquote(f)(&1))
  end

  @doc """
  get the width of the int or float type
  """
  def width(%__MODULE__{} = type) do
    cond do
      integer?(type) -> mlirIntegerTypeGetWidth(type)
      float?(type) -> mlirFloatTypeGetWidth(type)
    end
    |> Native.to_term()
  end

  defdelegate element_type(shaped_type), to: CAPI, as: :mlirShapedTypeGetElementType

  def llvm_pointer(opts \\ []),
    do: Deferred.from_opts(opts, fn ctx -> CAPI.mlirLLVMPointerTypeGet(ctx, opts[:address_space] || 0) end)
end
