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

  def normalize_dialect_name(to_upcase)
      when to_upcase in ~w{tosa gpu nvgpu omp nvvm llvm cf pdl rocdl spv amx amdgpu scf acc},
      do: String.upcase(to_upcase)

  def normalize_dialect_name("memref"), do: "MemRef"
  def normalize_dialect_name("emitc"), do: "EmitC"
  def normalize_dialect_name("arm_sve"), do: "ArmSVE"
  def normalize_dialect_name("x86vector"), do: "X86Vector"
  def normalize_dialect_name("ml_program"), do: "MLProgram"
  def normalize_dialect_name("pdl_interp"), do: "PDLInterp"
  def normalize_dialect_name(other), do: other |> Macro.camelize()

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
