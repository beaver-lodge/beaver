defmodule Beaver.MLIR.NIF do
  # This will persitent the nif library path so we can later open it
  Module.register_attribute(__MODULE__, :load_from, accumulate: false, persist: true)
  use Rustler, otp_app: :beaver_mlir, crate: "mlir_nif"

  # When your NIF is loaded, it will override this function.
  def add(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def load_from_path() do
    {:beaver_mlir, path} = @load_from
    path
  end
end
