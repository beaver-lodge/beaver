defmodule Beaver.MLIR.Type do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI

  def get(string, opts \\ [])

  def get(string, opts) when is_binary(string) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirTypeParseGet(ctx, MLIR.StringRef.create(string))
  end

  def equal?(a, b) do
    CAPI.mlirTypeEqual(a, b) |> Beaver.Native.to_term()
  end

  def function(inputs, results, opts \\ []) do
    num_inputs = length(inputs)
    num_results = length(results)
    inputs = inputs |> CAPI.MlirType.array()
    results = results |> CAPI.MlirType.array()
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirFunctionTypeGet(ctx, num_inputs, inputs, num_results, results)
  end

  def ranked_tensor(shape, element_type, encoding \\ nil)

  def ranked_tensor(
        shape,
        %MLIR.CAPI.MlirType{} = element_type,
        nil
      )
      when is_list(shape) do
    ranked_tensor(shape, element_type, CAPI.mlirAttributeGetNull())
  end

  def ranked_tensor(
        shape,
        %MLIR.CAPI.MlirType{} = element_type,
        encoding
      )
      when is_list(shape) do
    rank = length(shape)

    shape = shape |> Beaver.Native.I64.array()

    CAPI.mlirAttributeGetNull()
    CAPI.mlirRankedTensorTypeGet(rank, shape, element_type, encoding)
  end

  def unranked_tensor(%MLIR.CAPI.MlirType{} = element_type) do
    CAPI.mlirUnrankedTensorTypeGet(element_type)
  end

  def complex(%MLIR.CAPI.MlirType{} = element_type) do
    CAPI.mlirComplexTypeGet(element_type)
  end

  def memref(
        shape,
        %MLIR.CAPI.MlirType{} = element_type,
        opts \\ [layout: nil, memory_space: nil]
      )
      when is_list(shape) do
    rank = length(shape)

    shape = shape |> Beaver.Native.I64.array()

    default_null = CAPI.mlirAttributeGetNull()
    layout = Keyword.get(opts, :layout) || default_null
    memory_space = Keyword.get(opts, :memory_space) || default_null

    CAPI.mlirMemRefTypeGet(element_type, rank, shape, layout, memory_space)
  end

  def vector(shape, element_type) when is_list(shape) do
    rank = length(shape)
    shape = shape |> Beaver.Native.I64.array()
    CAPI.mlirVectorTypeGet(rank, shape, element_type)
  end

  def tuple(elements, opts \\ []) when is_list(elements) do
    num_elements = length(elements)
    elements = elements |> Beaver.Native.ISize.array()
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirTupleTypeGet(ctx, num_elements, elements)
  end

  def f16(opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirF16TypeGet(ctx)
  end

  def f32(opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirF32TypeGet(ctx)
  end

  def f64(opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirF64TypeGet(ctx)
  end

  def f(bitwidth, opts \\ []) when is_integer(bitwidth) do
    apply(__MODULE__, String.to_atom("f#{bitwidth}"), [opts])
  end

  def integer(bitwidth, opts \\ [signed: false]) do
    signed = Keyword.get(opts, :signed)
    ctx = MLIR.Managed.Context.from_opts(opts)

    if signed do
      CAPI.mlirIntegerTypeSignedGet(ctx, bitwidth)
    else
      CAPI.mlirIntegerTypeGet(ctx, bitwidth)
    end
  end

  def index(opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirIndexTypeGet(ctx)
  end

  defdelegate i(bitwidth, opts \\ []), to: __MODULE__, as: :integer

  for bitwidth <- [1, 8, 16, 32, 64, 128] do
    i_name = "i#{bitwidth}" |> String.to_atom()

    def unquote(i_name)() do
      apply(__MODULE__, :i, [unquote(bitwidth)])
    end
  end
end
