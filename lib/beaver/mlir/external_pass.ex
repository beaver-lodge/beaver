defmodule Beaver.MLIR.ExternalPass do
  @moduledoc false
  # Lower level API to work with MLIR's external pass (pass defined in C). Use Beaver.MLIR.Pass for idiomatic Erlang behavior.
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind,
    root_module: Beaver.MLIR.CAPI,
    forward_module: Beaver.Native

  defp op_name_from_persistent_attributes(pass_module) do
    op_name = pass_module.__info__(:attributes)[:root_op] || []
    op_name = op_name |> List.first()
    op_name || "builtin.module"
  end

  defp do_create(name, description, op, run) do
    {:ok, pid} = GenServer.start_link(MLIR.Pass.Server, run: run)
    description = description |> MLIR.StringRef.create()
    op_name = op |> MLIR.StringRef.create()
    name = name |> MLIR.StringRef.create()
    argument = name

    ref =
      CAPI.beaver_raw_create_mlir_pass(
        name.ref,
        argument.ref,
        description.ref,
        op_name.ref,
        pid
      )
      |> Beaver.Native.check!()

    %MLIR.Pass{ref: ref, handler: pid}
  end

  @doc """
  Create a pass by passing a callback module
  """

  def create({name, op, run}) when is_bitstring(op) and is_function(run) do
    description = "beaver generated pass of #{Function.info(run) |> inspect}"
    do_create(name, description, op, run)
  end

  def create(pass_module) when is_atom(pass_module) do
    description = "beaver generated pass of #{pass_module}"
    op_name = op_name_from_persistent_attributes(pass_module)
    name = Atom.to_string(pass_module)
    do_create(name, description, op_name, &pass_module.run/1)
  end
end
