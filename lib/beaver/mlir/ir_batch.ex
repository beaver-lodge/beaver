defmodule Beaver.MLIR.IRBatch do
  @moduledoc """
  Bounded IR snapshots and edits, with one native submission per batch (up to
  256 operations/edits). The caller owns the IR for the duration of the call.
  Snapshots contain borrowed handles and must not outlive or race IR mutation.

  Edits support `{:attribute, op, name, attribute}` and
  `{:operand, op, index, value}`. Every edit is checked before mutation; edits
  cannot erase operations. Verification failure leaves the modified IR in place.
  """
  alias Beaver.{Execution, MLIR, Native}

  def read(operations, opts \\ []) do
    opts = Keyword.put_new(opts, :strategy, :auto)
    names = Keyword.get(opts, :attributes, []) |> Enum.map(&to_string/1)

    case operations do
      [] ->
        []

      [first | _] ->
        ctx = MLIR.context(first)
        records = Execution.run(ctx, :snapshot, {Enum.map(operations, & &1.ref), names}, opts)

        Enum.map(records, fn {op, name, operands, results, operand_types, result_types, attributes} ->
          %{
            operation: Native.check!(op),
            name: name,
            operands: Enum.map(operands, &Native.check!/1),
            results: Enum.map(results, &Native.check!/1),
            operand_types: Enum.map(operand_types, &Native.check!/1),
            result_types: Enum.map(result_types, &Native.check!/1),
            attributes: Map.new(Enum.zip(names, Enum.map(attributes, &Native.check!/1)))
          }
        end)
    end
  end

  def update(root, edits, opts \\ []) do
    opts = Keyword.put_new(opts, :strategy, :auto)
    root = MLIR.Operation.from_module(root)

    commands =
      Enum.map(edits, fn
        {:attribute, %MLIR.Operation{ref: op}, name, %MLIR.Attribute{ref: attr}} ->
          {:attribute, op, to_string(name), attr}

        {:operand, %MLIR.Operation{ref: op}, index, %MLIR.Value{ref: value}} ->
          {:operand, op, index, value}

        edit ->
          raise ArgumentError, "unsupported IR batch edit: #{inspect(edit)}"
      end)

    {valid, diagnostics} =
      Execution.run(MLIR.context(root), :update, {root.ref, commands, Keyword.get(opts, :verifier, true)}, opts)

    if valid, do: {:ok, root, diagnostics}, else: {:error, diagnostics}
  end
end
