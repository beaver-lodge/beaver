defmodule Beaver.MLIR.StringRef.CallbackImpl do
  alias Beaver.MLIR

  def create() do
    raise "TODO: String Callback"
  end

  def handle_invoke(:string_ref_callback, [string_ref, _user_data_opaque_ptr], nil) do
    {:pass, MLIR.StringRef.extract(string_ref)}
  end

  def handle_invoke(:string_ref_callback, [string_ref, _user_data_opaque_ptr], state) do
    {:pass, state <> MLIR.StringRef.extract(string_ref)}
  end

  def collect_and_destroy(closure) do
    collected = closure |> Exotic.Closure.state()
    closure |> Exotic.Closure.destroy()
    collected
  end
end
