defmodule CharmTest do
  @moduledoc false
  use ExUnit.Case
  use Beaver

  defmodule CharmTest do
  end

  test "valid syntax" do
    Intermediator.SSA.extract(MyBoolean)
    |> Beaver.Charm.compile()
    |> Beaver.Charm.run()
  end
end
