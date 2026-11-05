alias Beaver.MLIR.{Type, TypeConverter}

defmodule Beaver.MLIR.TypeConverter.Callback do
  @moduledoc """
  Compatibility facade for a `Beaver.MLIR.TypeConverter` with one 1:1
  conversion callback. New code can use `Beaver.MLIR.TypeConverter` directly.
  """

  defstruct [:converter, :registration, :callback, timeout_ms: 30_000]

  @type callback_result() ::
          Type.t() | {:ok, Type.t()} | :declined | {:error, term()}
  @type t() :: %__MODULE__{
          converter: TypeConverter.t(),
          registration: reference(),
          callback: (Type.t() -> callback_result()),
          timeout_ms: non_neg_integer()
        }

  @spec create((Type.t() -> callback_result()), keyword()) :: t()
  def create(callback, opts \\ []) when is_function(callback, 1) do
    converter = TypeConverter.create(Keyword.put(opts, :conversion, callback))

    %__MODULE__{
      converter: converter,
      registration: converter.registration,
      callback: callback,
      timeout_ms: converter.timeout_ms
    }
  end

  @spec convert(t(), Type.t()) :: {:ok, Type.t()} | {:error, term()}
  def convert(%__MODULE__{converter: converter}, type),
    do: TypeConverter.convert(converter, type)

  @spec destroy(t()) :: :ok
  def destroy(%__MODULE__{converter: converter}), do: TypeConverter.destroy(converter)
end
