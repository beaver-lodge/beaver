alias Beaver.MLIR.Operation
alias Beaver.Deferred

defmodule Beaver.MLIR.Module do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  @behaviour Access

  @doc """
  Create a MLIR module by parsing string.
  """
  def create(str, opts \\ []) when is_binary(str) do
    Deferred.from_opts(
      opts,
      fn ctx ->
        {module, diagnostics} =
          Beaver.Execution.run(ctx, :parse, {MLIR.StringRef.create(str).ref})

        if MLIR.null?(module), do: {:error, diagnostics}, else: {:ok, module}
      end
    )
  end

  def create!(str, opts \\ []) when is_binary(str) do
    res =
      Deferred.from_opts(opts, fn ctx -> Deferred.resolve(create(str, opts), ctx) end)

    case res do
      %Deferred{} -> raise ArgumentError, "calling a bang function to parse module must be eager"
      {:error, diagnostics} -> raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "fail to parse module")
      {:ok, module} -> MLIR.verify!(module)
    end
  end

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  defdelegate destroy(module), to: CAPI, as: :mlirModuleDestroy

  @doc """
  Gets the body of the module, i.e. the only block it contains.
  """
  defdelegate body(module), to: CAPI, as: :mlirModuleGetBody
  defdelegate from_operation(op), to: CAPI, as: :mlirModuleFromOperation
  defdelegate empty(location), to: CAPI, as: :mlirModuleCreateEmpty

  @impl Access
  def fetch(module, attribute),
    do: Operation.fetch(Operation.from_module(module), attribute)

  @impl Access
  def get_and_update(module, attribute, function) do
    {attr, _} =
      Operation.get_and_update(Operation.from_module(module), attribute, function)

    {attr, module}
  end

  @impl Access
  def pop(module, attribute) do
    {attr, _} = Operation.pop(Operation.from_module(module), attribute)
    {attr, module}
  end
end
