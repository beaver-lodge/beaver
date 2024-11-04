defmodule Beaver.Printer do
  @moduledoc """
  This module provides a way to run MLIR CAPI with a string callback and user data.
  """
  alias Beaver.MLIR

  use Kinda.ResourceKind, forward_module: Beaver.Native

  @doc false
  def create() do
    Beaver.Native.forward(__MODULE__, :make, [])
    |> then(&{&1, Beaver.Native.forward(__MODULE__, :opaque_ptr, [&1])})
  end

  @doc false
  def callback() do
    %MLIR.StringCallback{ref: MLIR.CAPI.beaver_raw_string_printer_callback()}
  end

  @doc false
  def flush(sp) do
    MLIR.CAPI.beaver_raw_string_printer_flush(sp)
  end

  @doc """
  Run the given function with a string callback and user data.
  """
  def run(f) when is_function(f, 2) do
    {sp, user_data} = create()

    try do
      {f.(callback(), user_data), flush(sp)}
    rescue
      e ->
        flush(sp)
        reraise e, __STACKTRACE__
    end
  end
end
