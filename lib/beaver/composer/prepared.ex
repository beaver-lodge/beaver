defmodule Beaver.Composer.Prepared do
  @moduledoc """
  An immutable, context-bound BEAM pipeline template. Callback resolution and nesting checks are reused; each run
  constructs, initializes and destroys its own pass state and native managers. The context must outlive every run.
  Separate runs may execute concurrently on independently owned IR.
  """
  @enforce_keys [:context, :root, :stages]
  defstruct [:context, :root, :stages]

  def run(plan, ir, opts \\ []), do: Beaver.Composer.Beam.run_prepared(plan, ir, opts)

  def run!(plan, ir, opts \\ []), do: Beaver.Composer.unwrap_result!(run(plan, ir, opts))

  def profile(plan, ir, opts \\ []),
    do: Beaver.Execution.Profile.capture(fn profile -> run(plan, ir, Keyword.put(opts, :profile, profile)) end)
end
