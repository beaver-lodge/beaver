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
    Application.ensure_all_started(:beaver_capi)
    :ets.match(__MODULE__, {dialect, :"$1"}) |> List.flatten()
  end

  @less_used ~w{nvgpu
  gpu
  pdl_interp
  x86vector
  vector
  omp
  emitc
  sparse_tensor
  amdgpu
  async
  llvm
  transform
  ml_program
  amx
  arm_neon
  spv
  math
  quant
  arm_sve
  rocdl
  acc
  shape
  nvvm}

  @doc """
  Get dialects registered, if it is dev env in Mix, less used dialects will not be returned. Pass option dialects(full: true) to get all dialects.
  """
  def dialects(opts \\ [full: false]) do
    full = Keyword.get(opts, :full, true)
    Application.ensure_all_started(:beaver_capi)

    all_dialects =
      for [dialect, _] <- :ets.match(__MODULE__, {:"$1", :"$2"}) do
        dialect
      end
      |> Enum.uniq()

    case {full, Mix.env()} do
      {true, _} ->
        all_dialects

      {_, :prod} ->
        all_dialects

      _ ->
        all_dialects |> Enum.reject(fn x -> x in @less_used end)
    end
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
