defmodule Beaver.MLIR.Bytecode do
  @moduledoc """
  Reads and writes versioned MLIR bytecode.

  `:desired_emit_version` asks MLIR to emit a specific bytecode version. MLIR
  returns an error instead of silently emitting another version when the
  requested version cannot be honored.
  """

  alias Beaver.MLIR
  import MLIR.CAPI

  @type write_option :: {:desired_emit_version, integer() | nil}

  @spec write(MLIR.Module.t() | MLIR.Operation.t(), [write_option()]) ::
          {:ok, binary()} | {:error, :unsupported_bytecode_version}
  def write(module_or_operation, opts \\ []) do
    operation = MLIR.Operation.from_module(module_or_operation)
    config = mlirBytecodeWriterConfigCreate()

    try do
      case Keyword.get(opts, :desired_emit_version) do
        nil ->
          :ok

        version when is_integer(version) ->
          mlirBytecodeWriterConfigDesiredEmitVersion(config, version)
      end

      {result, bytecode} =
        Beaver.Printer.run(fn callback, user_data ->
          mlirOperationWriteBytecodeWithConfig(operation, config, callback, user_data)
        end)

      if MLIR.LogicalResult.success?(result) do
        {:ok, bytecode}
      else
        {:error, :unsupported_bytecode_version}
      end
    after
      mlirBytecodeWriterConfigDestroy(config)
    end
  end

  @spec write!(MLIR.Module.t() | MLIR.Operation.t(), [write_option()]) :: binary()
  def write!(module_or_operation, opts \\ []) do
    case write(module_or_operation, opts) do
      {:ok, bytecode} ->
        bytecode

      {:error, :unsupported_bytecode_version} ->
        raise ArgumentError, "MLIR could not honor the requested bytecode emit version"
    end
  end

  @spec read(binary(), keyword()) :: {:ok, MLIR.Module.t()} | {:error, MLIR.diagnostics()}
  defdelegate read(bytecode, opts \\ []), to: MLIR.Module, as: :create

  @spec read!(binary(), keyword()) :: MLIR.Module.t()
  defdelegate read!(bytecode, opts \\ []), to: MLIR.Module, as: :create!
end
