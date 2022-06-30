defmodule Beaver.Mlir.ExecutionEngine.Complex do
  def struct_fields(:f64) do
    [
      allocated: :ptr,
      aligned: :ptr,
      offset: :i64
    ]
  end
end
