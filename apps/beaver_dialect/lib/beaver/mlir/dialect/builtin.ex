defmodule Beaver.MLIR.Dialect.Builtin do
  alias Beaver.MLIR.Dialect

  use Beaver.MLIR.Dialect.Generator,
    dialect: "builtin",
    ops: Dialect.Registry.ops("builtin") |> Enum.reject(fn x -> x in ~w{module} end)

  defmodule Module do
    @behaviour Beaver.DSL.Op.Prototype

    @impl true
    def op_name() do
      "func.func"
    end
  end

  defmacro module(_call, do: _block) do
    raise "TODO: support module with symbol"
  end

  @doc """
  Macro to create a module and insert ops into its body. region/1 shouldn't be called because region of one block will be created.
  """
  defmacro module(do: block) do
    quote do
      location = Beaver.MLIR.Managed.Location.get()
      module = Beaver.MLIR.CAPI.mlirModuleCreateEmpty(location)
      module_body_block = Beaver.MLIR.CAPI.mlirModuleGetBody(module)

      Beaver.MLIR.Block.under(module_body_block, fn ->
        unquote(block)
      end)

      module
    end
  end
end
