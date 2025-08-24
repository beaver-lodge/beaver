defmodule ODSDumpTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke
  test "lookup" do
    assert {:ok,
            %{
              "attributes" => _,
              "operands" => _,
              "results" => _
            }} = MLIR.ODS.Dump.lookup("affine.for")

    assert {:error, "fail to found ods dump of \"???\""} = MLIR.ODS.Dump.lookup("???")
  end
end
