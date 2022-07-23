defmodule Beaver.MLIR.ExternalPass do
  @doc """
  Lower level API to work with MLIR's external pass (pass defined in C). Use Beaver.MLIR.Pass for idiomatic Erlang behavior.
  """
  defstruct external: nil
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI

  @doc """
  Create a pass by passing a callback module
  """
  def create(pass_module, op_name \\ "") do
    description = MLIR.StringRef.create("beaver generated pass of #{pass_module}")
    op_name = op_name |> MLIR.StringRef.create()
    name = Atom.to_string(pass_module) |> MLIR.StringRef.create()

    argument =
      Module.split(pass_module) |> List.last() |> Macro.underscore() |> MLIR.StringRef.create()

    {:ok, pid} = GenServer.start_link(MLIR.Pass.Server, pass_module: pass_module)

    ref =
      CAPI.beaver_raw_create_mlir_pass(
        name.ref,
        argument.ref,
        description.ref,
        op_name.ref,
        pid
      )
      |> CAPI.check!()

    %CAPI.MlirPass{ref: ref, handler: pid}
  end
end
