defmodule Beaver.Triton do
  @moduledoc """
  Registers Triton's core dialects and passes in a Beaver MLIR context.

  Once registered, a context can parse and inspect Triton IR (`tt`, `ttgir`,
  `ttng`, `ttinstrument`, `gluon` dialects), and Triton passes can be driven
  by name through `Beaver.Composer` pipelines.

  Requires a Beaver native build linked against the Triton core prebuilt
  (build with `BEAVER_TRITON_PREBUILT_DIR` set); otherwise the calls raise.
  """

  alias Beaver.MLIR

  @doc """
  Registers Triton passes and dialects, then loads the dialects on `context`.
  """
  @spec register(MLIR.Context.t()) :: :ok
  def register(%MLIR.Context{} = context) do
    register_passes()

    unless MLIR.CAPI.beaver_raw_triton_register_dialects(context.ref) do
      raise ArgumentError,
            "Beaver was built without Triton support; rebuild the native " <>
              "library with BEAVER_TRITON_PREBUILT_DIR set"
    end

    :ok
  end

  @doc """
  Registers Triton's passes in the global MLIR pass registry.

  Idempotent per process; safe to call multiple times.
  """
  @spec register_passes() :: :ok
  def register_passes do
    unless :persistent_term.get({__MODULE__, :passes_registered}, false) do
      unless MLIR.CAPI.beaver_raw_triton_register_passes() do
        raise ArgumentError,
              "Beaver was built without Triton support; rebuild the native " <>
                "library with BEAVER_TRITON_PREBUILT_DIR set"
      end

      :persistent_term.put({__MODULE__, :passes_registered}, true)
    end

    :ok
  end
end
