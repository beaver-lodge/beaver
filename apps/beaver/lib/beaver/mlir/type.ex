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
    CAPI.mlirTypeEqual(a, b) |> Exotic.Value.extract()
  end

  def function(inputs, results, opts \\ []) do
    num_inputs = length(inputs)
    num_results = length(results)
    inputs = inputs |> Exotic.Value.Array.get() |> Exotic.Value.get_ptr()
    results = results |> Exotic.Value.Array.get() |> Exotic.Value.get_ptr()
    CAPI.mlirFunctionTypeGet(num_inputs, inputs, num_results, results, opts)
  end

  def ranked_tensor(
        shape,
        %MLIR.CAPI.MlirType{} = element_type,
        encoding \\ Exotic.Value.Ptr.null()
      )
      when is_list(shape) do
    rank = length(shape)

    shape = shape |> Exotic.Value.Array.from_list({:i, 64}) |> Exotic.Value.get_ptr()

    CAPI.mlirRankedTensorTypeGet(rank, shape, element_type, encoding)
  end

  def memref(
        shape,
        %MLIR.CAPI.MlirType{} = element_type,
        opts \\ [layout: Exotic.Value.Ptr.null(), memory_space: Exotic.Value.Ptr.null()]
      )
      when is_list(shape) do
    rank = length(shape)

    shape = shape |> Exotic.Value.Array.from_list({:i, 64}) |> Exotic.Value.get_ptr()

    [layout: layout, memory_space: memory_space] =
      for k <- [:layout, :memory_space] do
        {k, Keyword.get(opts, k, Exotic.Value.Ptr.null())}
      end

    CAPI.mlirMemRefTypeGet(element_type, rank, shape, layout, memory_space)
  end

  def vector(shape, element_type) when is_list(shape) do
    rank = length(shape)
    shape = shape |> Exotic.Value.Array.from_list({:i, 64}) |> Exotic.Value.get_ptr()
    CAPI.mlirVectorTypeGet(rank, shape, element_type)
  end

  def tuple(elements, opts \\ []) when is_list(elements) do
    num_elements = length(elements)
    elements = elements |> Exotic.Value.Array.from_list() |> Exotic.Value.get_ptr()
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirTupleTypeGet(ctx, num_elements, elements)
  end

  def f32(opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirF32TypeGet(ctx)
  end

  def f(bitwidth, opts \\ []) when is_integer(bitwidth) do
    apply(__MODULE__, String.to_atom("f#{bitwidth}"), [opts])
  end

  defdelegate i(bitwidth, opts \\ []), to: __MODULE__, as: :integer

  for bitwidth <- [1, 8, 16, 32, 64, 128] do
    i_name = "i#{bitwidth}" |> String.to_atom()

    def unquote(i_name)() do
      apply(__MODULE__, :i, [unquote(bitwidth)])
    end
  end

  def to_string(type) do
    MLIR.StringRef.to_string(type, CAPI, :mlirTypePrint)
  end
end
