defmodule Beaver.MLIR.Sigils do
  alias Beaver.MLIR

  @doc """
  Create a module with global MLIR context or process MLIR context if registered.
  ## Examples
      iex> %MLIR.Module{} = ~m\"""
      ...> module {
      ...>   func.func @add(%arg0 : i32, %arg1 : i32) -> i32 attributes { llvm.emit_c_interface } {
      ...>     %res = arith.addi %arg0, %arg1 : i32
      ...>     return %res : i32
      ...>   }
      ...> }
      ...> \""" |> MLIR.Operation.verify!()
  """
  def sigil_m(string, []) do
    &MLIR.Module.create(&1, string)
  end

  @doc """
  Create an attribute with global MLIR context or process MLIR context if registered.
  You might add a modifier to it as a shortcut to annotate the type
  ## Examples

      iex> Attribute.equal?(Attribute.float(Type.f(32), 0.0), ~a{0.0}f32)
      true
      iex> ~a{1 : i32} |> MLIR.to_string()
      "1 : i32"
  """
  def sigil_a(string, []), do: MLIR.Attribute.get(string)

  def sigil_a(string, modifier) do
    modifier = modifier |> List.to_string()
    MLIR.Attribute.get("#{string} : #{modifier}")
  end

  @doc """
  Create a type with global MLIR context or process MLIR context if registered.
  You might add a modifier to it as a shortcut to make it a higher order type.
  ## Examples

      iex> Type.equal?(Type.unranked_tensor(Type.f32()), ~t{tensor<*xf32>})
      true
      iex> Type.equal?(Type.complex(Type.f32()), ~t<f32>complex)
      true
  """

  def sigil_t(string, []), do: MLIR.Type.get(string)

  def sigil_t(string, modifier) do
    modifier = modifier |> List.to_string()
    MLIR.Type.get("#{modifier}<#{string}>")
  end
end
