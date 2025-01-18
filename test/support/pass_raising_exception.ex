defmodule PassRaisingException do
  @moduledoc false
  use Beaver.MLIR.Pass, on: "func.func"

  def run(_op, _state) do
    raise "exception in pass run"
  end
end
