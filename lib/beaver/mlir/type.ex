alias Beaver.MLIR.{CAPI, StringRef}
alias Beaver.{Deferred, Native}

defmodule Beaver.MLIR.Type do
  @moduledoc """
  MLIR type parsing and construction, including integers, floats, index/none,
  tensors, vectors, memrefs and function types. Constructors accept a context
  in `:ctx` or return a deferred value; checked composite constructors return
  `{:ok, type}` or `{:error, diagnostics}`, with eager bang variants.
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
    {t, diagnostics} = apply(CAPI, getter, [ctx, loc | args])

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
    case apply(__MODULE__, cb, args) do
      %Deferred{} -> raise ArgumentError, "calling a bang function to compose a type must be eager"
      {:ok, t} -> t
      {:error, msg} -> raise ArgumentError, msg
    end
  end

  for {name, shaped?, getter} <- [
        {:ranked_tensor, true, :mlirRankedTensorTypeGetCheckedWithDiagnostics},
        {:unranked_tensor, false, :mlirUnrankedTensorTypeGetCheckedWithDiagnostics},
        {:memref, true, :mlirMemRefTypeGetCheckedWithDiagnostics},
        {:unranked_memref, false, :mlirUnrankedMemRefTypeGetCheckedWithDiagnostics},
        {:vector, true, :mlirVectorTypeGetCheckedWithDiagnostics}
      ] do
    leading = Macro.generate_arguments(if(shaped?, do: 1, else: 0), __MODULE__)
    shape = List.first(leading)
    guard = if shaped?, do: quote(do: is_list(unquote(shape))), else: true
    defaults = if name == :memref, do: [layout: nil, memory_space: nil], else: []

    if name == :vector do
      @doc """
      Get a vector type or deferred type value.

      ## Examples
          iex> ctx = MLIR.Context.create()
          iex> MLIR.Type.vector!([1, 2, 3], MLIR.Type.i32(ctx: ctx)) |> MLIR.to_string()
          "vector<1x2x3xi32>"
          iex> MLIR.Context.destroy(ctx)
      """
    end

    def unquote(name)(unquote_splicing(leading), element_type, opts \\ unquote(defaults))

    def unquote(name)(unquote_splicing(leading), %Deferred{} = element_type, opts) do
      Deferred.from_opts(opts, fn ctx ->
        unquote(name)(unquote_splicing(leading), Deferred.resolve(element_type, ctx), Keyword.put(opts, :ctx, ctx))
      end)
    end

    def unquote(name)(unquote_splicing(leading), %__MODULE__{} = element_type, opts) when unquote(guard) do
      ctx = composite_context(element_type, opts)
      args = composite_arguments(unquote(name), unquote(shape), element_type, ctx, opts)
      checked_composite_type(ctx, unquote(getter), args, opts)
    end

    def unquote(:"#{name}!")(unquote_splicing(leading), element_type, opts \\ []),
      do: bang_composite_type(unquote(name), [unquote_splicing(leading), element_type, opts])
  end

  defp composite_arguments(:unranked_tensor, _, type, _, _), do: [type]

  defp composite_arguments(:unranked_memref, _, type, ctx, opts),
    do: [type, composite_attribute(opts, :memory_space, ctx)]

  defp composite_arguments(kind, shape, type, ctx, opts) do
    rank = length(shape)
    shape = if kind == :vector, do: shape, else: Enum.map(shape, &MLIR.ShapedType.to_dynamic_magic_number(&1, :size))
    shape = Native.array(shape, Native.I64)

    case kind do
      :vector ->
        [rank, shape, type]

      :ranked_tensor ->
        [rank, shape, type, composite_attribute(opts, :encoding, ctx)]

      :memref ->
        layout = %MLIR.Attribute{} = composite_attribute(opts, :layout, ctx)
        space = %MLIR.Attribute{} = composite_attribute(opts, :memory_space, ctx)
        [type, rank, shape, layout, space]
    end
  end

  defp composite_attribute(opts, key, ctx), do: (opts[key] || mlirAttributeGetNull()) |> Deferred.resolve(ctx)

  def complex(%Deferred{} = element_type), do: Deferred.defer(&complex(Deferred.resolve(element_type, &1)))
  def complex(%__MODULE__{} = element_type), do: mlirComplexTypeGet(element_type)

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
