defmodule Beaver.MLIR.CAPI.Support do
  @deprecated "Use Beaver.MLIR.CAPI"
  @moduledoc false
  use Exotic.Library
  @path "libMLIRSupport.dylib"

  defmodule LogicalResult do
    use Exotic.Type.Struct, fields: [value: :i8]
  end

  defmodule TypeIDAllocator do
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule TypeID do
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end
  def load!(), do: Exotic.load!(__MODULE__, Beaver.MLIR.CAPI)
end
