defmodule Beaver.MLIR.Pass do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """

  alias Beaver.MLIR

  use Kinda.ResourceKind,
    fields: [handler: nil],
    forward_module: Beaver.Native

  @callback run(MLIR.Operation.t()) :: :ok | :error

  defmacro __using__(opts) do
    quote do
      @behaviour MLIR.Pass
      Module.register_attribute(__MODULE__, :root_op, persist: true, accumulate: false)
      @root_op Keyword.get(unquote(opts), :on, "builtin.module")
    end
  end
end
