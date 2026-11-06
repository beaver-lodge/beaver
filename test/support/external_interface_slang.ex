defmodule ExternalInterfaceSlang do
  @moduledoc false
  use Beaver.Slang, name: "external_interface_test"

  def pure_effects(_operation), do: :pure
  def write_effects(_operation), do: [:write]
  def speculatability(_operation), do: :speculatable

  defop pure(),
    interfaces: [
      memory_effects: &__MODULE__.pure_effects/1,
      conditionally_speculatable: &__MODULE__.speculatability/1
    ]

  defop write(), interfaces: [memory_effects: &__MODULE__.write_effects/1]

  defop detached()
end
