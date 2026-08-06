defmodule Beaver.MLIR.ConversionTarget do
  @moduledoc """
  An owning, callback-safe MLIR dialect conversion target.

  Static and dynamic legality rules are composed by MLIR. Dynamic callbacks
  run in the process that registered them while the conversion itself runs on
  the context's native worker pool.
  """

  use Kinda.ResourceKind,
    raw_module: Beaver.MLIR.CAPI.Raw,
    codec: Beaver.Native,
    fields: [registration: nil, timeout_ms: 30_000]

  alias Beaver.MLIR

  @type legality() :: :legal | :illegal | :no_opinion | boolean()
  @type legality_callback() :: (MLIR.Operation.t() -> legality())

  @spec create(MLIR.Context.t(), keyword()) :: t()
  def create(%MLIR.Context{ref: context_ref}, opts \\ []) do
    case Keyword.keys(opts) -- [:timeout] do
      [] ->
        :ok

      unsupported ->
        raise ArgumentError, "unsupported ConversionTarget options: #{inspect(unsupported)}"
    end

    timeout_ms = Keyword.get(opts, :timeout, 30_000)

    unless is_integer(timeout_ms) and timeout_ms >= 0 do
      raise ArgumentError, ":timeout must be a non-negative integer"
    end

    {:managed_conversion_target, target_term, registration} =
      MLIR.CAPI.beaver_raw_conversion_target_create(context_ref)

    %__MODULE__{ref: ref} = Beaver.Native.check!(target_term)
    %__MODULE__{ref: ref, registration: registration, timeout_ms: timeout_ms}
  end

  @spec add_legal_op(t(), String.Chars.t()) :: t()
  def add_legal_op(%__MODULE__{} = target, name) do
    add_static(target, name, 0)
  end

  @spec add_illegal_op(t(), String.Chars.t()) :: t()
  def add_illegal_op(%__MODULE__{} = target, name) do
    add_static(target, name, 1)
  end

  @spec add_legal_dialect(t(), String.Chars.t()) :: t()
  def add_legal_dialect(%__MODULE__{} = target, name) do
    add_static(target, name, 2)
  end

  @spec add_illegal_dialect(t(), String.Chars.t()) :: t()
  def add_illegal_dialect(%__MODULE__{} = target, name) do
    add_static(target, name, 3)
  end

  defp add_static(%__MODULE__{} = target, name, kind) do
    :ok =
      MLIR.CAPI.beaver_raw_conversion_target_add_static(
        target.registration,
        to_string(name),
        kind
      )

    target
  end

  @spec add_dynamically_legal_op(t(), String.Chars.t(), legality_callback()) :: t()
  def add_dynamically_legal_op(%__MODULE__{} = target, name, callback)
      when is_function(callback, 1) do
    :ok =
      MLIR.CAPI.beaver_raw_conversion_target_add_dynamic_op(
        target.registration,
        to_string(name),
        callback,
        target.timeout_ms
      )

    target
  end

  @spec add_dynamically_legal_dialect(t(), String.Chars.t(), legality_callback()) :: t()
  def add_dynamically_legal_dialect(%__MODULE__{} = target, name, callback)
      when is_function(callback, 1) do
    :ok =
      MLIR.CAPI.beaver_raw_conversion_target_add_dynamic_dialect(
        target.registration,
        to_string(name),
        callback,
        target.timeout_ms
      )

    target
  end

  @spec mark_recursively_legal(t(), String.Chars.t(), legality_callback() | nil) :: t()
  def mark_recursively_legal(%__MODULE__{} = target, name, callback \\ nil)
      when is_nil(callback) or is_function(callback, 1) do
    :ok =
      MLIR.CAPI.beaver_raw_conversion_target_mark_recursively_legal(
        target.registration,
        to_string(name),
        callback,
        target.timeout_ms
      )

    target
  end

  @spec mark_unknown_dynamically_legal(t(), legality_callback()) :: t()
  def mark_unknown_dynamically_legal(%__MODULE__{} = target, callback)
      when is_function(callback, 1) do
    :ok =
      MLIR.CAPI.beaver_raw_conversion_target_mark_unknown_dynamic(
        target.registration,
        callback,
        target.timeout_ms
      )

    target
  end

  @spec destroy(t()) :: :ok
  def destroy(%__MODULE__{registration: registration}) do
    MLIR.CAPI.beaver_raw_conversion_target_destroy(registration)
  end

  @spec with(MLIR.Context.t(), keyword(), (t() -> result)) :: result when result: var
  def with(ctx, opts \\ [], fun) when is_function(fun, 1) do
    target = create(ctx, opts)

    try do
      fun.(target)
    after
      destroy(target)
    end
  end
end
