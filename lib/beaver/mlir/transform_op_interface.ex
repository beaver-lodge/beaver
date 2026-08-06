defmodule Beaver.MLIR.TransformOpInterface do
  @moduledoc """
  Implements MLIR's `TransformOpInterface` for dynamic operations.

  `apply/4` callbacks receive the transform operation, a borrowed
  `Beaver.MLIR.TransformRewriter`, borrowed results storage, and borrowed
  transform state. They return `:ok`, `{:ok, mappings}`, a silenceable failure,
  or a definite failure. Result mappings associate an operation result (or its
  integer index) with `{:ops, operations}`, `{:values, values}`, or
  `{:params, attributes}`.

      {:ok, %{0 => {:ops, payload_operations}}}

  All borrowed values are valid only for the dynamic extent of the callback.
  Use `payload_ops/2`, `payload_values/2`, and `params/2` to inspect mappings,
  and `rewriter_base/1` to use the normal rewriter API.
  """

  alias Beaver.MLIR

  @type result_mapping() ::
          %{
            optional(non_neg_integer() | MLIR.Value.t()) =>
              {:ops, [MLIR.Operation.t()]}
              | {:values, [MLIR.Value.t()]}
              | {:params, [MLIR.Attribute.t()]}
          }
          | keyword()

  @type apply_result() ::
          :ok
          | {:ok, result_mapping() | nil}
          | :silenceable_failure
          | :definite_failure
          | {:error, :silenceable | :definite}

  @callback apply(
              MLIR.Operation.t(),
              MLIR.TransformRewriter.t(),
              MLIR.TransformResults.t(),
              MLIR.TransformState.t()
            ) :: apply_result()
  @callback allows_repeated_handle_operands?(MLIR.Operation.t()) :: boolean()
  @optional_callbacks allows_repeated_handle_operands?: 1

  @doc "Attaches a callback-backed transform operation interface model."
  @spec attach(MLIR.Context.t(), String.t(), module() | keyword() | map(), keyword()) ::
          MLIR.ExternalInterface.Attachment.t()
  def attach(context, operation_name, implementation, opts \\ []) do
    callbacks = callbacks!(implementation)
    MLIR.ExternalInterface.attach(context, operation_name, :transform_op, callbacks, opts)
  end

  @doc false
  def callbacks!(implementation) when is_atom(implementation) do
    unless function_exported?(implementation, :apply, 4) do
      raise ArgumentError, "#{inspect(implementation)} must implement apply/4"
    end

    callbacks = %{apply: &implementation.apply/4}

    if function_exported?(implementation, :allows_repeated_handle_operands?, 1) do
      Map.put(
        callbacks,
        :allows_repeated_handle_operands,
        &implementation.allows_repeated_handle_operands?/1
      )
    else
      callbacks
    end
  end

  def callbacks!(implementation) when is_list(implementation) do
    implementation
    |> Map.new()
    |> callbacks!()
  end

  def callbacks!(%{} = implementation) do
    apply_callback = Map.get(implementation, :apply)

    unless is_function(apply_callback, 4) do
      raise ArgumentError, "transform interface requires an :apply callback with arity 4"
    end

    repeated =
      Map.get(implementation, :allows_repeated_handle_operands) ||
        Map.get(implementation, :allows_repeated_handle_operands?)

    unless is_nil(repeated) or is_function(repeated, 1) do
      raise ArgumentError,
            ":allows_repeated_handle_operands callback must have arity 1"
    end

    %{apply: apply_callback, allows_repeated_handle_operands: repeated}
  end

  def callbacks!(other),
    do: raise(ArgumentError, "invalid transform interface implementation: #{inspect(other)}")

  @doc "Returns payload operations mapped to a transform handle."
  @spec payload_ops(MLIR.TransformState.t(), MLIR.Value.t()) :: [MLIR.Operation.t()]
  def payload_ops(%MLIR.TransformState{} = state, %MLIR.Value{} = handle) do
    MLIR.CAPI.beaver_raw_transform_state_payload_ops(state.ref, handle.ref)
    |> Enum.map(&Beaver.Native.check!/1)
  end

  @doc "Returns payload values mapped to a transform handle."
  @spec payload_values(MLIR.TransformState.t(), MLIR.Value.t()) :: [MLIR.Value.t()]
  def payload_values(%MLIR.TransformState{} = state, %MLIR.Value{} = handle) do
    MLIR.CAPI.beaver_raw_transform_state_payload_values(state.ref, handle.ref)
    |> Enum.map(&Beaver.Native.check!/1)
  end

  @doc "Returns parameters mapped to a transform handle."
  @spec params(MLIR.TransformState.t(), MLIR.Value.t()) :: [MLIR.Attribute.t()]
  def params(%MLIR.TransformState{} = state, %MLIR.Value{} = handle) do
    MLIR.CAPI.beaver_raw_transform_state_params(state.ref, handle.ref)
    |> Enum.map(&Beaver.Native.check!/1)
  end

  @doc "Casts a borrowed transform rewriter to the normal rewriter base."
  @spec rewriter_base(MLIR.TransformRewriter.t()) :: MLIR.RewriterBase.t()
  def rewriter_base(%MLIR.TransformRewriter{} = rewriter) do
    MLIR.CAPI.mlirTransformRewriterAsBase(rewriter)
  end

  @doc false
  def set_results(%MLIR.TransformResults{} = results, %MLIR.Operation{} = operation, mappings) do
    mappings
    |> Enum.each(fn {result, mapping} ->
      set_result(results, result_value(operation, result), mapping)
    end)

    :ok
  end

  defp result_value(operation, index) when is_integer(index) and index >= 0,
    do: MLIR.Operation.result(operation, index)

  defp result_value(_operation, %MLIR.Value{} = value), do: value

  defp result_value(_operation, other),
    do: raise(ArgumentError, "invalid transform result key: #{inspect(other)}")

  defp set_result(results, result, {:ops, operations}) when is_list(operations) do
    set_result_list(
      :mlirTransformResultsSetOps,
      results,
      result,
      operations,
      MLIR.Operation,
      "payload operations"
    )
  end

  defp set_result(results, result, {:values, values}) when is_list(values) do
    set_result_list(
      :mlirTransformResultsSetValues,
      results,
      result,
      values,
      MLIR.Value,
      "payload values"
    )
  end

  defp set_result(results, result, {:params, attributes}) when is_list(attributes) do
    set_result_list(
      :mlirTransformResultsSetParams,
      results,
      result,
      attributes,
      MLIR.Attribute,
      "parameters"
    )
  end

  defp set_result(_results, _result, other),
    do: raise(ArgumentError, "invalid transform result mapping: #{inspect(other)}")

  defp set_result_list(function, results, result, values, module, label) do
    unless Enum.all?(values, &is_struct(&1, module)) do
      raise ArgumentError, "#{label} must contain only #{inspect(module)} values"
    end

    array = Beaver.Native.array(values, module, mut: true)
    apply(MLIR.CAPI, function, [results, result, length(values), array])
    :ok
  end
end
