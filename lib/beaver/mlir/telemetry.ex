defmodule Beaver.MLIR.Telemetry do
  @moduledoc false

  @prefix [:beaver, :mlir, :compilation]

  def emit(event_suffix, measurements, metadata, opts \\ []) do
    event = @prefix ++ event_suffix

    if function_exported?(:telemetry, :execute, 3) do
      apply(:telemetry, :execute, [event, measurements, metadata])
    end

    case Keyword.get(opts, :telemetry) do
      callback when is_function(callback, 3) -> callback.(event, measurements, metadata)
      _ -> :ok
    end

    :ok
  end
end
