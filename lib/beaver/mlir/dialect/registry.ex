defmodule Beaver.MLIR.Dialect.Registry do
  @moduledoc """
  This module defines functions to query MLIR dialect registry.
  """

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

  def normalize_dialect_name(to_upcase)
      when to_upcase in ~w{tosa gpu nvgpu omp nvvm llvm cf pdl rocdl spirv amx amdgpu scf acc dlti},
      do: String.upcase(to_upcase)

  def normalize_dialect_name("memref"), do: "MemRef"
  def normalize_dialect_name("emitc"), do: "EmitC"
  def normalize_dialect_name("arm_sve"), do: "ArmSVE"
  def normalize_dialect_name("arm_sme"), do: "ArmSME"
  def normalize_dialect_name("x86vector"), do: "X86Vector"
  def normalize_dialect_name("ml_program"), do: "MLProgram"
  def normalize_dialect_name("pdl_interp"), do: "PDLInterp"
  def normalize_dialect_name("irdl"), do: "IRDL"
  def normalize_dialect_name("mpi"), do: "MPI"
  def normalize_dialect_name("xegpu"), do: "XeGPU"
  def normalize_dialect_name(other), do: other |> Macro.camelize()

  defp unwrap_ctx_then(opts, cb) do
    if ctx = opts[:ctx] do
      %MLIR.Context{ref: ref} = ctx

      cb.(ref)
    else
      ctx = %MLIR.Context{ref: ref} = MLIR.Context.create()
      ret = cb.(ref)
      MLIR.Context.destroy(ctx)
      ret
    end
  end

  def ops(dialect, opts \\ []) do
    unwrap_ctx_then(opts, fn ref ->
      Beaver.Native.check!(CAPI.beaver_raw_registered_ops(ref))
      |> Stream.filter(&String.starts_with?(&1, "#{dialect}."))
      |> Enum.map(&String.trim_leading(&1, "#{dialect}."))
    end)
  end

  @doc """
  Get dialects registered
  """
  def dialects(opts \\ []) do
    unwrap_ctx_then(opts, fn ref ->
      CAPI.beaver_raw_registered_dialects(ref)
      |> Beaver.Native.check!()
      |> Enum.uniq()
      |> Enum.sort()
    end)
  end
end
