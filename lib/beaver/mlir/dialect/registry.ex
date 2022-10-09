defmodule Beaver.MLIR.Dialect.Registry do
  require Beaver.MLIR.CAPI
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR

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
  end

  def normalize_dialect_name(to_upcase)
      when to_upcase in ~w{tosa gpu nvgpu omp nvvm llvm cf pdl rocdl spirv amx amdgpu scf acc dlti},
      do: String.upcase(to_upcase)

  def normalize_dialect_name("memref"), do: "MemRef"
  def normalize_dialect_name("emitc"), do: "EmitC"
  def normalize_dialect_name("arm_sve"), do: "ArmSVE"
  def normalize_dialect_name("x86vector"), do: "X86Vector"
  def normalize_dialect_name("ml_program"), do: "MLProgram"
  def normalize_dialect_name("pdl_interp"), do: "PDLInterp"
  def normalize_dialect_name(other), do: other |> Macro.camelize()

  def ops(dialect) do
    Beaver.Native.check!(
      CAPI.beaver_raw_registered_ops_of_dialect(MLIR.StringRef.create(dialect).ref)
    )
    |> Enum.map(&List.to_string/1)
  end

  @doc """
  Get dialects registered, if it is dev/test env with config key :skip_dialects of app :beaver configured,
  these dialects will not be returned (usually to speedup the compilation). Pass option dialects(full: true) to get all dialects anyway.
  """
  def dialects(opts \\ [full: false]) do
    full = Keyword.get(opts, :full, false)

    all_dialects =
      CAPI.beaver_raw_registered_dialects()
      |> Enum.map(&List.to_string/1)
      |> Beaver.Native.check!()
      |> Enum.uniq()
      |> Enum.sort()

    if full do
      all_dialects
    else
      skip_dialects = Application.get_env(:beaver, :skip_dialects, [])
      all_dialects |> Enum.reject(fn x -> x in skip_dialects end)
    end
  end
end
