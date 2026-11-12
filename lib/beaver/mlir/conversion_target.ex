alias Beaver.MLIR.{CAPI, Context}

defmodule Beaver.MLIR.ConversionTarget do
  @moduledoc """
  An owning, callback-safe MLIR dialect conversion target.

  Static and dynamic legality rules are composed by MLIR. Dynamic callbacks run in the process that registered them
  while the conversion itself runs on the context's native worker pool.
  """

  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Beaver.Native, fields: [registration: nil, timeout_ms: 30_000]

  alias Beaver.MLIR

  @type legality() :: :legal | :illegal | :no_opinion | boolean()
  @type legality_callback() :: (MLIR.Operation.t() -> legality())

  @spec create(Context.t(), keyword()) :: t()
  def create(%Context{ref: context_ref}, opts \\ []) do
    case Keyword.keys(opts) -- [:timeout] do
      [] -> :ok
      unsupported -> raise ArgumentError, "unsupported ConversionTarget options: #{inspect(unsupported)}"
    end

    timeout_ms = Beaver.Execution.callback_timeout!(opts)
    {:managed_conversion_target, target_term, registration} = CAPI.beaver_raw_conversion_target_create(context_ref)
    %__MODULE__{ref: ref} = Beaver.Native.check!(target_term)
    %__MODULE__{ref: ref, registration: registration, timeout_ms: timeout_ms}
  end

  for {name, kind} <- Enum.with_index(~w(add_legal_op add_illegal_op add_legal_dialect add_illegal_dialect)a) do
    @spec unquote(name)(t(), String.Chars.t()) :: t()
    def unquote(name)(%__MODULE__{} = target, name) do
      :ok = CAPI.beaver_raw_conversion_target_add_static(target.registration, to_string(name), unquote(kind))
      target
    end
  end

  for {name, nif} <- [
        add_dynamically_legal_op: :beaver_raw_conversion_target_add_dynamic_op,
        add_dynamically_legal_dialect: :beaver_raw_conversion_target_add_dynamic_dialect
      ] do
    @spec unquote(name)(t(), String.Chars.t(), legality_callback()) :: t()
    def unquote(name)(%__MODULE__{} = target, name, callback) when is_function(callback, 1),
      do: add_dynamic(target, name, callback, unquote(nif))
  end

  defp add_dynamic(target, name, callback, nif) do
    :ok = apply(CAPI, nif, [target.registration, to_string(name), callback, target.timeout_ms])
    target
  end

  @spec mark_recursively_legal(t(), String.Chars.t(), legality_callback() | nil) :: t()
  def mark_recursively_legal(%__MODULE__{} = target, name, callback \\ nil)
      when is_nil(callback) or is_function(callback, 1) do
    add_dynamic(target, name, callback, :beaver_raw_conversion_target_mark_recursively_legal)
  end

  @spec mark_unknown_dynamically_legal(t(), legality_callback()) :: t()
  def mark_unknown_dynamically_legal(%__MODULE__{} = target, callback) when is_function(callback, 1) do
    :ok = CAPI.beaver_raw_conversion_target_mark_unknown_dynamic(target.registration, callback, target.timeout_ms)

    target
  end

  @spec destroy(t()) :: :ok
  def destroy(%__MODULE__{registration: registration}), do: CAPI.beaver_raw_conversion_target_destroy(registration)

  @spec with(Context.t(), keyword(), (t() -> result)) :: result when result: var
  def with(ctx, opts \\ [], fun) when is_function(fun, 1),
    do: Beaver.Native.with_resource(create(ctx, opts), &destroy/1, fun)
end
