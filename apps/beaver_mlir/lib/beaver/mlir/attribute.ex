defmodule Beaver.MLIR.Attribute do
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR

  def is_null(jit) do
    jit
    |> Exotic.Value.fetch(MLIR.CAPI.MlirAttribute, :ptr)
    |> Exotic.Value.extract() == 0
  end

  def get(attr_str) when is_binary(attr_str) do
    ctx = MLIR.Managed.Context.get()
    attr = MLIR.StringRef.create(attr_str)
    attr = CAPI.mlirAttributeParseGet(ctx, attr)

    if is_null(attr) do
      raise "fail to parse attribute: #{attr_str}"
    end

    attr
  end

  def equal?(a, b) do
    CAPI.mlirAttributeEqual(a, b) |> Exotic.Value.extract()
  end

  def unit() do
    ctx = MLIR.Managed.Context.get()
    CAPI.mlirUnitAttrGet(ctx)
  end

  def to_string(attr) do
    MLIR.StringRef.to_string(attr, CAPI, :mlirAttributePrint)
  end

  def float(type, value, opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirFloatAttrDoubleGet(ctx, type, value)
  end

  def dense_elements(elements, shaped_type) when is_list(elements) do
    num_elements = length(elements)
    elements = elements |> Exotic.Value.Array.from_list() |> Exotic.Value.get_ptr()
    dense_elements(shaped_type, num_elements, elements)
  end

  for {:function_signature,
       [
         f = %Exotic.CodeGen.Function{
           name: name,
           args: args,
           ret: {:type_def, Beaver.MLIR.CAPI.MlirAttribute}
         }
       ]} <-
        MLIR.CAPI.__info__(:attributes) do
    name_str = Atom.to_string(name)
    is_type_get = name_str |> String.ends_with?("AttrGet")

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
end
