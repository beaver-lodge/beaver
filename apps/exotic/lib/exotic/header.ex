defmodule Exotic.Header do
  # enforced keys are provided by user
  @enforce_keys [:file, :search_paths]
  defstruct file: nil,
            module_name: nil,
            search_paths: [],
            functions: [],
            structs: [],
            type_defs: []

  def parse(h = %__MODULE__{module_name: module_name}) when not is_nil(module_name) do
    parent = self()

    {:ok, :dirty} =
      System.trap_signal(:sigchld, :dirty, fn ->
        send(parent, {:nif, :ok})
        :ok
      end)

    # This create sub process so we trap the signal
    declarations = Exotic.CodeGen.parse_header(h)

    receive do
      {:nif, :ok} ->
        :ok = System.untrap_signal(:sigchld, :dirty)
    end

    declarations
  end
end
