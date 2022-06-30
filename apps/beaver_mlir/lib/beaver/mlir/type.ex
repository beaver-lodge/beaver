defmodule Beaver.MLIR.Type do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  def get(string, opts \\ [])

  def get(string, opts) when is_binary(string) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirTypeParseGet(ctx, MLIR.StringRef.create(string))
  end

  def equal?(a, b) do
    CAPI.mlirTypeEqual(a, b) |> Exotic.Value.extract()
  end

  def ranked_tensor(
        shape,
        %MLIR.CAPI.MlirType{} = element_type,
        encoding \\ Exotic.Value.Ptr.null()
      )
      when is_list(shape) do
    rank = length(shape)

    shape = shape |> Exotic.Value.Array.from_list() |> Exotic.Value.get_ptr()

    ranked_tensor(rank, shape, element_type, encoding)
  end

  def memref(
        shape,
        %MLIR.CAPI.MlirType{} = element_type,
        opts \\ [layout: Exotic.Value.Ptr.null(), memory_space: Exotic.Value.Ptr.null()]
      )
      when is_list(shape) do
    rank = length(shape)

    shape = shape |> Exotic.Value.Array.from_list() |> Exotic.Value.get_ptr()

    [layout: layout, memory_space: memory_space] =
      for k <- [:layout, :memory_space] do
        {k, Keyword.get(opts, k, Exotic.Value.Ptr.null())}
      end

    CAPI.mlirMemRefTypeGet(element_type, rank, shape, layout, memory_space)
  end

  for {:function_signature,
       [
         f = %Exotic.CodeGen.Function{
           name: name,
           args: args,
           ret: {:type_def, Beaver.MLIR.CAPI.MlirType}
         }
       ]} <-
        MLIR.CAPI.__info__(:attributes) do
    name_str = Atom.to_string(name)
    is_type_get = name_str |> String.ends_with?("TypeGet")

    if is_type_get do
      "mlir" <> generated_func_name = name_str
      generated_func_name = generated_func_name |> String.slice(0..-8) |> Macro.underscore()
      generated_func_name = generated_func_name |> String.to_atom()
      IO.inspect(f)

      arg_names =
        for {arg_name, _} <- args do
          arg_name |> Atom.to_string() |> Macro.underscore()
        end
        |> Enum.join(", ")

      @doc """
      generated from
      ```
      #{inspect(f, pretty: true)}
      ```
      """
      case args do
        [{_ctx, {:type_def, Beaver.MLIR.CAPI.MlirContext}} | rest_args] ->
          IO.inspect("#{generated_func_name}(#{arg_names})")

          args =
            for {arg_name, _} <- rest_args do
              arg_name = arg_name |> Atom.to_string() |> Macro.underscore() |> String.to_atom()
              {arg_name, [], nil}
            end

          def unquote(generated_func_name)(unquote_splicing(args), opts \\ []) do
            ctx = MLIR.Managed.Context.from_opts(opts)
            apply(Beaver.MLIR.CAPI, unquote(name), [ctx, unquote_splicing(args)])
          end

        args ->
          args =
            for {arg_name, _} <- args do
              arg_name = arg_name |> Atom.to_string() |> Macro.underscore() |> String.to_atom()
              {arg_name, [], nil}
            end

          def unquote(generated_func_name)(unquote_splicing(args)) do
            apply(Beaver.MLIR.CAPI, unquote(name), [unquote_splicing(args)])
          end
      end
    end
  end

  def f(bitwidth, opts \\ []) when is_integer(bitwidth) do
    apply(__MODULE__, String.to_atom("f#{bitwidth}"), [opts])
  end

  def to_string(type) do
    MLIR.StringRef.to_string(type, CAPI, :mlirTypePrint)
  end
end
