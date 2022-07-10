defmodule Beaver.MLIR.Dialect.Registry do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  use GenServer

  require Beaver.MLIR.CAPI

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

  defp pre_process(op_name) do
    case op_name do
      "tilezero" ->
        "x86_tilezero"

      "tile_zero" ->
        "tile_zero"

      _ ->
        op_name
    end
  end

  def normalize_op_name(op_name) do
    pre_process(op_name) |> String.replace(".", "_") |> Macro.underscore() |> String.to_atom()
  end

  def op_module_name(op_name) do
    pre_process(op_name)
    |> String.split(".")
    |> Enum.join("_")
    |> Macro.camelize()
    |> String.to_atom()

    # |> Enum.map(&Macro.camelize/1)
    # |> Module.concat()
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

  defp do_ops(dialect, query: false) do
    :ets.match(__MODULE__, {dialect, :"$1"}) |> List.flatten()
  end

  defp do_ops(dialect, query: true) do
    for {^dialect, o} <- query_ops() do
      o
    end
  end

  def ops(dialect, opts \\ [query: false]) do
    do_ops(dialect, opts)
  end

  @doc """
  Get dialects registered, if it is dev/test env with config key :skip_dialects of app :beaver configured,
  these dialects will not be returned (usually to speedup the compilation). Pass option dialects(full: true) to get all dialects anyway.
  """
  def dialects(opts \\ [full: false, query: false]) do
    query = Keyword.get(opts, :query, false)

    full = Keyword.get(opts, :full, true)

    all_dialects =
      if query do
        query_ops() |> Enum.map(fn {d, _o} -> d end)
      else
        for [dialect, _] <- :ets.match(__MODULE__, {:"$1", :"$2"}) do
          dialect
        end
      end
      |> Enum.uniq()

    if full do
      all_dialects
    else
      skip_dialects = Application.get_env(:beaver, :skip_dialects, [])
      all_dialects |> Enum.reject(fn x -> x in skip_dialects end)
    end
  end

  defp query_ops() do
    lib = MLIR.CAPI.load!()
    context = Exotic.call!(lib, :mlirContextCreate, [])
    Exotic.call!(lib, :mlirRegisterAllDialects, [context])

    num_op =
      Exotic.call!(lib, :beaverGetNumRegisteredOperations, [context])
      |> Exotic.Value.extract()

    for i <- 0..(num_op - 1)//1 do
      op_name =
        Exotic.call!(lib, :beaverGetRegisteredOperationName, [context, Exotic.Value.get(:i64, i)])

      dialect_name =
        op_name
        |> then(fn name ->
          Exotic.call!(lib, :beaverRegisteredOperationNameGetDialectName, [name])
        end)
        |> MLIR.StringRef.extract()

      op_name =
        op_name
        |> then(fn name ->
          Exotic.call!(lib, :beaverRegisteredOperationNameGetOpName, [name])
        end)
        |> MLIR.StringRef.extract()

      {dialect_name, op_name}
  end
end
