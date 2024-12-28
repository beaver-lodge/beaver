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

  defp escape_dynamic(:dynamic), do: MLIR.CAPI.mlirShapedTypeGetDynamicStrideOrOffset()

  defp escape_dynamic(dim), do: dim

  def ranked_tensor(shape, element_type, encoding \\ nil)

  def ranked_tensor(
        shape,
        f,
        encoding
      )
      when is_function(f, 1) do
    &ranked_tensor(shape, f.(&1), encoding)
  end

  def ranked_tensor(
        shape,
        %__MODULE__{} = element_type,
        nil
      )
      when is_list(shape) do
    ranked_tensor(shape, element_type, mlirAttributeGetNull())
  end

  def ranked_tensor(
        shape,
        %__MODULE__{} = element_type,
        encoding
      )
      when is_list(shape) do
    rank = length(shape)

    shape =
      shape
      |> Enum.map(&escape_dynamic/1)
      |> Beaver.Native.array(Beaver.Native.I64)

    mlirRankedTensorTypeGet(rank, shape, element_type, encoding)
  end

  def unranked_tensor(element_type)
      when is_function(element_type, 1) do
    &unranked_tensor(element_type.(&1))
  end

  def unranked_tensor(%__MODULE__{} = element_type) do
    mlirUnrankedTensorTypeGet(element_type)
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
    rank = length(shape)

    shape = shape |> Enum.map(&escape_dynamic/1) |> Beaver.Native.array(Beaver.Native.I64)

    default_null = mlirAttributeGetNull()
    layout = Keyword.get(opts, :layout) || default_null
    memory_space = Keyword.get(opts, :memory_space) || default_null

    mlirMemRefTypeGet(element_type, rank, shape, layout, memory_space)
  end

  @doc """
  Get a vector type creator.

  ## Examples
      iex> ctx = MLIR.Context.create()
      iex> MLIR.Type.vector([1, 2, 3], MLIR.Type.i32).(ctx) |> MLIR.to_string()
      "vector<1x2x3xi32>"
      iex> MLIR.Context.destroy(ctx)
  """

  def vector(shape, element_type) when is_function(element_type, 1) do
    &vector(shape, element_type.(&1))
  end

  def vector(shape, %__MODULE__{} = element_type) when is_list(shape) do
    rank = length(shape)
    shape = shape |> Beaver.Native.array(Beaver.Native.I64)
    mlirVectorTypeGet(rank, shape, element_type)
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

  def f16(opts \\ []) do
    Beaver.Deferred.from_opts(opts, &mlirF16TypeGet/1)
  end

  def f32(opts \\ []) do
    Beaver.Deferred.from_opts(opts, &mlirF32TypeGet/1)
  end

  def f64(opts \\ []) do
    Beaver.Deferred.from_opts(opts, &mlirF64TypeGet/1)
  end

  def float(bitwidth, opts \\ []) when is_integer(bitwidth) do
    apply(__MODULE__, String.to_atom("f#{bitwidth}"), [opts])
  end

  defdelegate f(bitwidth, opts \\ []), to: __MODULE__, as: :float

  def integer(bitwidth, opts \\ [signed: false]) do
    signed = Keyword.get(opts, :signed)

    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        if signed do
          mlirIntegerTypeSignedGet(ctx, bitwidth)
        else
          mlirIntegerTypeGet(ctx, bitwidth)
        end
      end
    )
  end

  def index(opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      &mlirIndexTypeGet(&1)
    )
  end

  def none(opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      &mlirNoneTypeGet(&1)
    )
  end

  defdelegate i(bitwidth, opts \\ []), to: __MODULE__, as: :integer

  for bitwidth <- [1, 8, 16, 32, 64, 128] do
    i_name = "i#{bitwidth}" |> String.to_atom()

    def unquote(i_name)(opts \\ []) do
      apply(__MODULE__, :i, [unquote(bitwidth), opts])
    end
  end

  for {f, "mlirTypeIsA" <> type_name, 1} <-
        Beaver.MLIR.CAPI.__info__(:functions)
        |> Enum.map(fn {f, a} -> {f, Atom.to_string(f), a} end) do
    helper_name = type_name |> Macro.underscore() |> String.replace("mem_ref", "memref")

    def unquote(:"#{helper_name}?")(%MLIR.Type{} = t) do
      unquote(f)(t) |> Beaver.Native.to_term()
    end
  end
end
