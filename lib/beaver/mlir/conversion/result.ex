defmodule Beaver.MLIR.Conversion.Result do
  @moduledoc """
  A conversion outcome with mutation and legalization information.

  `legalizable` (analysis mode) and `unlegalized` (partial mode) contain copied
  name/location/preorder-index snapshots of original operations, never dangling
  operation handles. Partial conversion may stop at an explicitly illegal op;
  `unlegalized_complete?` is then false. Full conversion does not collect a
  legalization inventory. Analysis mode always rolls back speculative changes.

  Mutation reporting covers edits made through the conversion rewriter. Native
  rollback cannot undo arbitrary side effects performed by user callbacks.
  """
  defstruct [
    :ir,
    :mode,
    :status,
    :changed?,
    :rolled_back?,
    :unlegalized_complete?,
    legalizable: [],
    unlegalized: [],
    diagnostics: []
  ]

  @type t :: %__MODULE__{status: :succeeded | :failed | :callback_failed, mode: :full | :partial | :analysis}

  @doc false
  def decode(json, mode, ir, diagnostics, callback_failure) do
    fields = JSON.decode!(json)

    status =
      cond do
        callback_failure != nil -> :callback_failed
        fields["success"] -> :succeeded
        true -> :failed
      end

    %__MODULE__{
      ir: ir,
      mode: mode,
      status: status,
      changed?: fields["changed"],
      rolled_back?: fields["rolled_back"],
      unlegalized_complete?: fields["unlegalized_complete"],
      legalizable: fields["legalizable"],
      unlegalized: fields["unlegalized"],
      diagnostics: diagnostics
    }
  end
end
