defmodule Beaver.Sigils do
  @moduledoc """
  Sigils return explicit `Beaver.Deferred` values that parse MLIR elements in a context.
  """
  alias Beaver.MLIR

  @doc """
  Create a deferred module value.
  ## Examples

      iex> ctx = MLIR.Context.create()
      iex> %MLIR.Module{} = ~m\"""
      ...> module {
      ...>   func.func @add(%arg0 : i32, %arg1 : i32) -> i32 attributes { llvm.emit_c_interface } {
      ...>     %res = arith.addi %arg0, %arg1 : i32
      ...>     return %res : i32
      ...>   }
      ...> }
      ...> \""" |> Beaver.Deferred.resolve(ctx) |> MLIR.verify!()
      iex> MLIR.Context.destroy(ctx)
  """
  def sigil_m(string, []), do: Beaver.Deferred.defer(&MLIR.Module.create!(string, ctx: &1))

  @doc """
  Create a deferred attribute value.

  Add a modifier to it as a shortcut to annotate the type. The `s` modifier
  creates a string attribute directly, without parsing MLIR text.
  ## Examples

      iex> ctx = MLIR.Context.create()
      iex> MLIR.equal?(Beaver.Deferred.resolve(Attribute.float(Type.f(32), 0.0), ctx), Beaver.Deferred.resolve(~a{0.0}f32, ctx))
      true
      iex> ~a{1 : i32} |> Beaver.Deferred.resolve(ctx) |> MLIR.to_string()
      "1 : i32"
      iex> value = ~s(string with "quotes")
      iex> MLIR.equal?(Beaver.Deferred.resolve(~a/\#{value}/s, ctx), Beaver.Deferred.resolve(Attribute.string(value), ctx))
      true
      iex> MLIR.Context.destroy(ctx)
  """
  def sigil_a(string, []), do: MLIR.Attribute.get(string)
  def sigil_a(string, [?s]), do: MLIR.Attribute.string(string)

  def sigil_a(string, modifier) do
    modifier = modifier |> List.to_string()
    MLIR.Attribute.get("#{string} : #{modifier}")
  end

  @doc """
  Create an SSA builder for an operation whose name is known at runtime.

  Prefer an explicitly declared Elixir dialect function when the operation
  name is stable. This sigil is intended for genuinely dynamic operation names.

  A bare operation sigil creates an operation without arguments. Call the
  returned builder with `.()` to pass operands or attributes:

      mlir do
        ~o/\#{operation_name}/ >>> []
        ~o/\#{operation_name}/.(operand, attribute: value) >>> result_type
      end
  """
  def sigil_o(operation_name, []), do: operation_builder(operation_name)

  defp operation_builder(operation_name) do
    fn %Beaver.SSA{evaluator: evaluator} = ssa when is_function(evaluator, 1) ->
      evaluator.(%Beaver.SSA{ssa | op: operation_name})
    end
  end

  @doc """
  Create a deferred type value.

  Add a modifier to it as a shortcut to make it a higher order type.
  ## Examples

      iex> ctx = MLIR.Context.create()
      iex> MLIR.equal?(Type.unranked_tensor!(Type.f32(ctx: ctx)), Beaver.Deferred.resolve(~t{tensor<*xf32>}, ctx))
      true
      iex> MLIR.equal?(Type.unranked_tensor!(Type.f32(ctx: ctx)), ~t{tensor<*xf32>})
      true
      iex> MLIR.equal?(Beaver.Deferred.resolve(Type.complex(Type.f32()), ctx), Beaver.Deferred.resolve(~t<f32>complex, ctx))
      true
      iex> MLIR.equal?(Type.complex(Type.f32()), Beaver.Deferred.resolve(~t<f32>complex, ctx))
      true
      iex> MLIR.Context.destroy(ctx)
  """

  def sigil_t(string, []), do: MLIR.Type.get(string)

  def sigil_t(string, modifier) do
    modifier = modifier |> List.to_string()
    MLIR.Type.get("#{modifier}<#{string}>")
  end
end
