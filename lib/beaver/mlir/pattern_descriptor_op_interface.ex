defmodule Beaver.MLIR.PatternDescriptorOpInterface do
  @moduledoc """
  Implements Transform dialect's `PatternDescriptorOpInterface` for dynamic
  operations.

  The callback receives a borrowed `Beaver.MLIR.RewritePatternSet` and adds
  Beaver rewrite patterns to it. An optional state-aware callback also receives
  a borrowed `Beaver.MLIR.TransformState`. Borrowed handles must not escape the
  callback.
  """

  alias Beaver.MLIR

  @callback populate_patterns(MLIR.Operation.t(), MLIR.RewritePatternSet.t()) :: any()
  @callback populate_patterns_with_state(
              MLIR.Operation.t(),
              MLIR.RewritePatternSet.t(),
              MLIR.TransformState.t()
            ) :: any()
  @optional_callbacks populate_patterns_with_state: 3

  @doc "Attaches a callback-backed pattern descriptor interface model."
  @spec attach(MLIR.Context.t(), String.t(), module() | keyword() | map(), keyword()) ::
          MLIR.ExternalInterface.Attachment.t()
  def attach(context, operation_name, implementation, opts \\ []) do
    callbacks = callbacks!(implementation)
    MLIR.ExternalInterface.attach(context, operation_name, :pattern_descriptor, callbacks, opts)
  end

  @doc false
  def callbacks!(implementation) when is_atom(implementation) do
    unless function_exported?(implementation, :populate_patterns, 2) do
      raise ArgumentError, "#{inspect(implementation)} must implement populate_patterns/2"
    end

    callbacks = %{populate_patterns: &implementation.populate_patterns/2}

    if function_exported?(implementation, :populate_patterns_with_state, 3) do
      Map.put(
        callbacks,
        :populate_patterns_with_state,
        &implementation.populate_patterns_with_state/3
      )
    else
      callbacks
    end
  end

  def callbacks!(implementation) when is_list(implementation),
    do: implementation |> Map.new() |> callbacks!()

  def callbacks!(%{} = implementation) do
    populate = Map.get(implementation, :populate_patterns)
    with_state = Map.get(implementation, :populate_patterns_with_state)

    unless is_function(populate, 2) do
      raise ArgumentError,
            "pattern descriptor requires a :populate_patterns callback with arity 2"
    end

    unless is_nil(with_state) or is_function(with_state, 3) do
      raise ArgumentError, ":populate_patterns_with_state callback must have arity 3"
    end

    %{populate_patterns: populate, populate_patterns_with_state: with_state}
  end

  def callbacks!(other),
    do: raise(ArgumentError, "invalid pattern descriptor implementation: #{inspect(other)}")
end
