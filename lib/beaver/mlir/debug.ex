defmodule Beaver.MLIR.Debug do
  @moduledoc """
  Configure MLIR global debug options and attach LLVM debug information so
  translated IR and JIT'd code can be traced back to Elixir source lines with
  gdb/lldb.

  Beaver records the Elixir `Macro.Env` location (`loc("/path/file.exs":line:col)`)
  on every operation it builds, and those locations survive lowering to the LLVM
  dialect. The MLIR-to-LLVM translation, however, only emits DWARF `DILocation`
  metadata when each `llvm.func` carries a `DISubprogramAttr` scope.

  `attach_llvm_scopes!/1` runs MLIR's upstream
  `ensure-debug-info-scope-on-llvm-func` pass, which attaches a `DISubprogramAttr`
  (and a compile unit derived from the module location) to every `llvm.func`.
  After it runs:

    * `mlir-translate --mlir-to-llvmir` emits `!dbg` and `!DILocation` metadata
      pointing at the original `.exs` files and lines;
    * `Beaver.MLIR.ExecutionEngine.create!(module, debug_info: true)` produces a
      JIT object whose line table is registered with gdb (MLIR's execution
      engine enables its GDB notification listener by default), so
      `break file.exs:line` works while attached to the BEAM process.

  The pass emits line tables only (`DIEmissionKind::LineTablesOnly`) by design:
  it is a convenient way to step line-by-line or get a backtrace with line
  numbers, not a replacement for frontends emitting complete debug info.
  """
  alias Beaver.Composer
  alias Beaver.MLIR
  import MLIR.CAPI

  @doc """
  Enable or disable global debugging.
  """
  def enable(enable \\ true) do
    mlirEnableGlobalDebug(enable)
  end

  def disable() do
    enable(false)
  end

  @doc """
  Check if global debugging is enabled.

  ## Examples

      iex> Beaver.MLIR.Debug.enabled?()
      false
  """
  def enabled?() do
    mlirIsGlobalDebugEnabled() |> Beaver.Native.to_term()
  end

  @doc """
  Set debug type(s).

  Note: Global debug must be enabled for any output to be produced.

  ## Examples

      iex> Beaver.MLIR.Debug.set_debug_type("pass-manager")
      :ok

      iex> Beaver.MLIR.Debug.set_debug_type(~w[pass-manager dialect-conversion])
      :ok
  """
  def set_debug_type(type) when is_binary(type) do
    type_str = MLIR.StringRef.create(type) |> MLIR.StringRef.data()
    mlirSetGlobalDebugType(type_str)
  end

  def set_debug_type(types) when is_list(types) do
    types
    |> Enum.map(&MLIR.StringRef.create/1)
    |> Beaver.Native.array(MLIR.StringRef)
    |> beaverSetGlobalDebugTypes(length(types))
  end

  @doc """
  Check if a specific debug type is currently enabled.

  ## Examples

      iex> Beaver.MLIR.Debug.is_current_debug_type?("pass-manager")
      true
  """
  def is_current_debug_type?(type) when is_binary(type) do
    type_str = MLIR.StringRef.create(type) |> MLIR.StringRef.data()
    mlirIsCurrentDebugType(type_str) |> Beaver.Native.to_term()
  end

  @doc """
  Attach a `DISubprogramAttr` debug scope to every `llvm.func` in the module.

  The pass is idempotent: functions that already carry a subprogram scope are
  left untouched. The attached scopes reference the Elixir file and line that
  each `llvm.func` was originally built from, so the MLIR-to-LLVM translation
  can emit `DILocation` metadata for it.

  Requires the module to be in the LLVM dialect (the pass fails on other
  modules).
  """
  @spec attach_llvm_scopes!(Composer.t() | MLIR.Module.t() | MLIR.Operation.t()) ::
          MLIR.Module.t() | MLIR.Operation.t()
  def attach_llvm_scopes!(composer_or_op) do
    pass = MLIR.CAPI.mlirCreateLLVMDIScopeForLLVMFuncOpPass()
    composer_or_op |> Composer.append(pass) |> Composer.run!()
  end
end
