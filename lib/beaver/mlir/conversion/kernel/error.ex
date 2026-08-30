defmodule Beaver.MLIR.Conversion.Kernel.Error do
  @moduledoc """
  A structured, fail-closed compiler-kernel contract error.

  `code` is stable protocol data. `message` is intended for humans and may be
  refined without changing the protocol. `details` contains bounded values
  that identify the rejected manifest or artifact.
  """

  defexception [:code, :message, details: %{}]

  @type code() ::
          :invalid_manifest
          | :unsupported_schema
          | :abi_mismatch
          | :beaver_revision_mismatch
          | :llvm_revision_mismatch
          | :dialect_schema_mismatch
          | :runtime_abi_mismatch
          | :target_mismatch
          | :capability_missing
          | :artifact_unreadable
          | :artifact_digest_mismatch

  @type t() :: %__MODULE__{
          code: code(),
          message: String.t(),
          details: map()
        }

  @impl Exception
  def exception(opts) do
    code = Keyword.fetch!(opts, :code)
    details = Keyword.get(opts, :details, %{})
    message = Keyword.get(opts, :message, default_message(code, details))
    %__MODULE__{code: code, details: details, message: message}
  end

  defp default_message(code, details) do
    "compiler-kernel contract rejected #{code}: #{inspect(details, limit: 20)}"
  end
end
