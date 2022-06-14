defmodule Beaver.MLIR.Dialect.Registry do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  use GenServer

  def start_link([]) do
    GenServer.start_link(__MODULE__, [])
  end

  def init(init_arg) do
    :ets.new(__MODULE__, [:bag, :named_table, read_concurrency: true])

    for o <- query_ops() do
      :ets.insert(__MODULE__, o)
    end

    {:ok, init_arg}
  end

  def normalize_op_name(op_name) do
    op_name |> String.replace(".", "_") |> Macro.underscore() |> String.to_atom()
  end

  def ops(dialect) do
    Application.ensure_all_started(:beaver_mlir)
    :ets.match(__MODULE__, {dialect, :"$1"}) |> List.flatten()
  end

  def dialects() do
    Application.ensure_all_started(:beaver_mlir)

    for [dialect, _] <- :ets.match(__MODULE__, {:"$1", :"$2"}) do
      dialect
    end
    |> Enum.uniq()
  end

  defp query_ops() do
    context = MLIR.Managed.Context.get()

    num_op =
      CAPI.beaverGetNumRegisteredOperations(context)
      |> Exotic.Value.extract()

    for i <- 0..(num_op - 1)//1 do
      op_name = CAPI.beaverGetRegisteredOperationName(context, i)

      dialect_name =
        op_name
        |> CAPI.beaverRegisteredOperationNameGetDialectName()
        |> MLIR.StringRef.extract()

      op_name =
        op_name
        |> CAPI.beaverRegisteredOperationNameGetOpName()
        |> MLIR.StringRef.extract()

      {dialect_name, op_name}
    end
  end
end
