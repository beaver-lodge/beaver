defmodule CustomTraitSlang.Validity do
  @moduledoc false
end

defmodule CustomTraitSlang.RegionValidity do
  @moduledoc false
end

defmodule CustomTraitSlang do
  @moduledoc false
  use Beaver.Slang, name: "custom_trait_test"

  def verify(operation) do
    if Beaver.MLIR.Operation.name(operation) == "custom_trait_test.invalid" do
      {:error, :invalid_operation}
    else
      :ok
    end
  end

  def verify_regions(operation) do
    if Beaver.MLIR.Operation.name(operation) == "custom_trait_test.invalid_regions" do
      raise "invalid regions"
    else
      :ok
    end
  end

  defop valid(),
    traits: [{CustomTraitSlang.Validity, verify: &__MODULE__.verify/1}]

  defop also_valid(),
    traits: [{CustomTraitSlang.Validity, verify: &__MODULE__.verify/1}]

  defop invalid(),
    traits: [{CustomTraitSlang.Validity, verify: &__MODULE__.verify/1}]

  defop invalid_regions(),
    traits: [
      {CustomTraitSlang.RegionValidity, verify_regions: &__MODULE__.verify_regions/1}
    ]
end
