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
  import Beaver.MLIR.CAPI

  use Kinda.ResourceKind, forward_module: Beaver.Native

  def get(string, opts \\ [])

  def get(string, opts) when is_binary(string) do
    Beaver.Deferred.from_opts(opts, fn ctx ->
      t = mlirTypeParseGet(ctx, MLIR.StringRef.create(string))

      if MLIR.null?(t) do
        raise "fail to parse type: #{string}"
      end

      t
    end)
  end

  def function(inputs, results, opts \\ []) do
    inputs = List.wrap(inputs)
    results = List.wrap(results)
    num_inputs = length(inputs)
    num_results = length(results)

    Beaver.Deferred.from_opts(opts, fn ctx ->
      inputs =
        inputs |> Enum.map(&Beaver.Deferred.create(&1, ctx)) |> Beaver.Native.array(__MODULE__)

      results =
        results
        |> Enum.map(&Beaver.Deferred.create(&1, ctx))
        |> Beaver.Native.array(__MODULE__)

      mlirFunctionTypeGet(ctx, num_inputs, inputs, num_results, results)
    end)
  end

  defp checked_composite_type(ctx, getter, args, opts) do
    loc = opts[:loc] || MLIR.Location.unknown(ctx: ctx)
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
      f when is_function(f, 1) ->
        raise ArgumentError, "calling a bang function to compose a type must be eager"

      {:ok, t} ->
        t

      {:error, msg} ->
        raise ArgumentError, msg
    end
  end

  def ranked_tensor(shape, element_type, opts \\ [])

  def ranked_tensor(
        shape,
        f,
        opts
      )
      when is_function(f, 1) do
    &ranked_tensor(shape, f.(&1), opts)
  end

  def ranked_tensor(
        shape,
        %__MODULE__{} = element_type,
        opts
      )
      when is_list(shape) do
    ctx = MLIR.context(element_type)
    rank = length(shape)
    encoding = opts[:encoding] || mlirAttributeGetNull()

    shape =
      shape
      |> Enum.map(&__MODULE__.Shaped.to_dynamic_magic_number(&1, :size))
      |> Beaver.Native.array(Beaver.Native.I64)

    checked_composite_type(
      ctx,
      &mlirRankedTensorTypeGetCheckedWithDiagnostics/6,
      [rank, shape, element_type, encoding],
      opts
    )
  end

  def ranked_tensor!(shape, element_type, opts \\ []) do
    bang_composite_type(&ranked_tensor/3, [shape, element_type, opts])
  end

  def unranked_tensor(element_type, opts \\ [])

  def unranked_tensor(element_type, opts)
      when is_function(element_type, 1) do
    &unranked_tensor(element_type.(&1), opts)
  end

  def unranked_tensor(%__MODULE__{} = element_type, opts) do
    ctx = MLIR.context(element_type)

    checked_composite_type(
      ctx,
      &mlirUnrankedTensorTypeGetCheckedWithDiagnostics/3,
      [element_type],
      opts
    )
  end

  def unranked_tensor!(element_type, opts \\ []) do
    bang_composite_type(&unranked_tensor/2, [element_type, opts])
  end

  def unranked_memref(element_type, opts \\ [])

  def unranked_memref(element_type, opts)
      when is_function(element_type, 1) do
    &unranked_memref(element_type.(&1), opts)
  end

  def unranked_memref(%__MODULE__{} = element_type, opts) do
    ctx = MLIR.context(element_type)
    default_null = mlirAttributeGetNull()
    memory_space = (opts[:memory_space] || default_null) |> Beaver.Deferred.create(ctx)

    checked_composite_type(
      ctx,
      &mlirUnrankedMemRefTypeGetCheckedWithDiagnostics/4,
      [element_type, memory_space],
      opts
    )
  end

  def unranked_memref!(element_type, opts \\ []) do
    bang_composite_type(&unranked_memref/2, [element_type, opts])
  end

  def complex(element_type) when is_function(element_type, 1) do
    &complex(element_type.(&1))
  end

  def complex(%__MODULE__{} = element_type) do
    mlirComplexTypeGet(element_type)
  end

  def memref(
        shape,
        element_type,
        opts \\ [layout: nil, memory_space: nil]
      )

  def memref(shape, element_type, opts) when is_function(element_type, 1) do
    &memref(shape, element_type.(&1), opts)
  end

  def memref(
        shape,
        %__MODULE__{} = element_type,
        opts
      )
      when is_list(shape) do
    ctx = MLIR.context(element_type)
    rank = length(shape)

    shape =
      shape
      |> Enum.map(&__MODULE__.Shaped.to_dynamic_magic_number(&1, :size))
      |> Beaver.Native.array(Beaver.Native.I64)

    default_null = mlirAttributeGetNull()

    layout =
      %MLIR.Attribute{} = (opts[:layout] || default_null) |> Beaver.Deferred.create(ctx)

    memory_space =
      %MLIR.Attribute{} = (opts[:memory_space] || default_null) |> Beaver.Deferred.create(ctx)

    checked_composite_type(
      ctx,
      &mlirMemRefTypeGetCheckedWithDiagnostics/7,
      [element_type, rank, shape, layout, memory_space],
      opts
    )
  end

  def memref!(shape, %__MODULE__{} = element_type, opts \\ []) do
    bang_composite_type(&memref/3, [shape, element_type, opts])
  end

  @doc """
  Get a vector type creator.

  ## Examples
      iex> ctx = MLIR.Context.create()
      iex> MLIR.Type.vector!([1, 2, 3], MLIR.Type.i32(ctx: ctx)) |> MLIR.to_string()
      "vector<1x2x3xi32>"
      iex> MLIR.Context.destroy(ctx)
  """

  def vector(shape, element_type, opts \\ [])

  def vector(shape, element_type, opts) when is_function(element_type, 1) do
    &vector(shape, element_type.(&1), opts)
  end

  def vector(shape, %__MODULE__{} = element_type, opts) when is_list(shape) do
    ctx = MLIR.context(element_type)
    rank = length(shape)
    shape = shape |> Beaver.Native.array(Beaver.Native.I64)

    checked_composite_type(
      ctx,
      &mlirVectorTypeGetCheckedWithDiagnostics/5,
      [rank, shape, element_type],
      opts
    )
  end

  def vector!(shape, %__MODULE__{} = element_type, opts \\ []) do
    bang_composite_type(&vector/3, [shape, element_type, opts])
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
    Beaver.Deferred.from_opts(opts, fn ctx ->
      num_elements = length(elements)

      elements =
        elements
        |> Enum.map(&Beaver.Deferred.create(&1, ctx))
        |> Beaver.Native.array(__MODULE__)

      mlirTupleTypeGet(ctx, num_elements, elements)
    end)
  end

  def float(bitwidth, opts \\ []) when is_integer(bitwidth) do
    apply(__MODULE__, String.to_atom("f#{bitwidth}"), [opts])
  end

  defdelegate f(bitwidth, opts \\ []), to: __MODULE__, as: :float

  def opaque(dialect_namespace, type_data, opts \\ []) do
    Beaver.Deferred.from_opts(opts, fn ctx ->
      mlirOpaqueTypeGet(
        ctx,
        MLIR.StringRef.create(dialect_namespace),
        MLIR.StringRef.create(type_data)
      )
    end)
  end

  def integer(bitwidth, opts \\ [signed: false]) do
    signed = Keyword.get(opts, :signed)

    Beaver.Deferred.from_opts(
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
        Beaver.MLIR.CAPI.__info__(:functions)
        |> Enum.map(fn {f, a} -> {f, Atom.to_string(f), a} end) do
    type_name =
      type_name
      |> String.replace(~r"Type$", "")

    @doc """
    calls `Beaver.MLIR.CAPI.#{f}/1` to check if it is #{type_name} type.
    """
    helper_name =
      type_name
      |> Macro.underscore()
      |> String.replace("mem_ref", "memref")

    def unquote(:"#{helper_name}?")(%__MODULE__{} = t) do
      unquote(f)(t) |> Beaver.Native.to_term()
    end
  end

  for sign_type <- ~w{signless signed unsigned} do
    f = :"mlirIntegerTypeIs#{Macro.camelize(sign_type)}"

    @doc """
    calls `Beaver.MLIR.CAPI.#{f}/1` to check if it is a #{sign_type} integer.
    """

    def unquote(:"#{sign_type}?")(%__MODULE__{} = type) do
      unquote(f)(type) |> Beaver.Native.to_term()
    end
  end

  for {f, "mlir" <> type_name, 1} <-
        Beaver.MLIR.CAPI.__info__(:functions)
        |> Enum.map(fn {f, a} -> {f, Atom.to_string(f), a} end)
        |> Enum.filter(fn {_, type_name, _} ->
          String.ends_with?(type_name, "TypeGet") and
            not String.contains?(type_name, "Complex") and
            not String.contains?(type_name, "UnrankedTensor")
        end) do
    type_name =
      type_name
      |> String.trim_trailing("TypeGet")

    @doc """
    calls `Beaver.MLIR.CAPI.#{f}/1` to get #{type_name} type
    """
    helper_name =
      type_name
      |> Macro.underscore()
      |> String.replace("mem_ref", "memref")

    def unquote(:"#{helper_name}")(opts \\ []) do
      Beaver.Deferred.from_opts(
        opts,
        &unquote(f)(&1)
      )
    end
  end

  @doc """
  get the width of the int or float type
  """
  def width(%__MODULE__{} = type) do
    cond do
      integer?(type) ->
        mlirIntegerTypeGetWidth(type)

      float?(type) ->
        mlirFloatTypeGetWidth(type)
    end
    |> Beaver.Native.to_term()
  end

  defdelegate element_type(shaped_type), to: MLIR.CAPI, as: :mlirShapedTypeGetElementType

  def llvm_pointer(opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        MLIR.CAPI.mlirLLVMPointerTypeGet(ctx, opts[:address_space] || 0)
      end
    )
  end
end
