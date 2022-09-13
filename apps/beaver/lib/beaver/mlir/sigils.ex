defmodule Beaver.MLIR.Sigils do
  alias Beaver.MLIR

  @doc """
  Create an module creator.
  ## Examples

      iex> ctx = MLIR.Context.create()
      iex> %MLIR.Module{} = ~m\"""
      ...> module {
      ...>   func.func @add(%arg0 : i32, %arg1 : i32) -> i32 attributes { llvm.emit_c_interface } {
      ...>     %res = arith.addi %arg0, %arg1 : i32
      ...>     return %res : i32
      ...>   }
      ...> }
      ...> \""".(ctx) |> MLIR.Operation.verify!()
      iex> ctx |> MLIR.Context.destroy
  """
  def sigil_m(string, []) do
    &MLIR.Module.create(&1, string)
  end

  @doc """
  Create an attribute creator.
  You might add a modifier to it as a shortcut to annotate the type
  ## Examples

      iex> ctx = MLIR.Context.create()
      iex> Attribute.equal?(Attribute.float(Type.f(32), 0.0).(ctx), ~a{0.0}f32.(ctx))
      true
      iex> ~a{1 : i32}.(ctx) |> MLIR.to_string()
      "1 : i32"
      iex> ctx |> MLIR.Context.destroy
  """
  def sigil_a(string, []), do: MLIR.Attribute.get(string)

  def sigil_a(string, modifier) do
    modifier = modifier |> List.to_string()
    MLIR.Attribute.get("#{string} : #{modifier}")
  end

  @doc """
  Create an type creator.
  You might add a modifier to it as a shortcut to make it a higher order type.
  ## Examples

      iex> ctx = MLIR.Context.create()
      iex> Type.equal?(Type.unranked_tensor(Type.f32()).(ctx), ~t{tensor<*xf32>}.(ctx))
      true
      iex> Type.equal?(Type.complex(Type.f32()).(ctx), ~t<f32>complex.(ctx))
      true
      iex> ctx |> MLIR.Context.destroy
  """

  def sigil_t(string, []), do: MLIR.Type.get(string)

  def sigil_t(string, modifier) do
    modifier = modifier |> List.to_string()
    MLIR.Type.get("#{modifier}<#{string}>")
  end
end
