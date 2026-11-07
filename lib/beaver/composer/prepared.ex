defmodule Beaver.Composer.Prepared do
  @moduledoc """
  An immutable, context-bound BEAM pipeline template. Callback resolution and
  nesting checks are reused; each run constructs, initializes and destroys its
  own pass state and native managers. The context must outlive every run.
  Separate runs may execute concurrently on independently owned IR.
  """
  @enforce_keys [:context, :root, :stages]
  defstruct [:context, :root, :stages]
  require Logger

  def run(plan, ir, opts \\ []), do: Beaver.Composer.Beam.run_prepared(plan, ir, opts)

  def run!(plan, ir, opts \\ []) do
    case run(plan, ir, opts) do
      {:ok, ir, []} ->
        ir

      {:ok, ir, diagnostics} ->
        Logger.warning(Beaver.MLIR.Diagnostic.format(diagnostics, "Passes completed with diagnostics"))
        ir

      {:error, diagnostics} ->
        raise ArgumentError, Beaver.MLIR.Diagnostic.format(diagnostics)
    end
  end

  def profile(plan, ir, opts \\ []),
    do: Beaver.Execution.Profile.capture(fn profile -> run(plan, ir, Keyword.put(opts, :profile, profile)) end)
end
