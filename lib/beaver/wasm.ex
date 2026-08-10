defmodule Beaver.Wasm do
  @moduledoc """
  WebAssembly packaging for MLIR modules — the WASM analogue of the
  `gpu.binary` packaging.

  `package_binary!/2` lowers a module to LLVM IR (via the `ex` conversion
  plan and the standard arith/scf/cf-to-llvm passes), translates it in-process,
  compiles it to a wasm binary with the Zig wasm32 target, and returns a
  `Beaver.Wasm.Binary` carrying the bytes plus the imports/exports manifest
  parsed from the binary.

  This is the host-boundary step of the batata WASM preparation (Forgejo
  #37): the manifest names the WASI/import surface that a runtime must
  provide, and the binary is ready for a `:wasm` Shadow Wavefront evaluator.
  """

  import Beaver.MLIR.Conversion
  import Beaver.MLIR.Transform

  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.{Ex, Plan}
  alias Beaver.MLIR.Target.LLVMIR

  @doc """
  Packages an MLIR module into a `Beaver.Wasm.Binary`.

  Options:

    * `:target` — wasm target triple (default `"wasm32-wasi"`)
    * `:entry` — symbol to export (default `"main"`)
    * `:opt` — Zig optimization flag (default `"-O2"`)
    * `:nostdlib` — compile without libc/WASI imports (default `true`)
  """
  @spec package_binary!(MLIR.Module.t(), keyword()) :: Beaver.Wasm.Binary.t()
  def package_binary!(%MLIR.Module{} = module, opts \\ []) do
    target = Keyword.get(opts, :target, "wasm32-wasi")
    entry = Keyword.get(opts, :entry, "main")
    opt = Keyword.get(opts, :opt, "-O2")
    nostdlib = Keyword.get(opts, :nostdlib, true)

    llvm_ir = to_llvm_ir!(module)
    bytes = compile!(llvm_ir, target, entry, opt, nostdlib)

    %{Beaver.Wasm.Binary.parse(bytes) | target: target}
  end

  defp to_llvm_ir!(module) do
    llvm =
      module
      |> then(&if(has_ex_ops?(&1), do: Plan.run!(Ex.plan(), &1), else: &1))
      |> canonicalize()
      |> convert_scf_to_cf()
      |> convert_to_llvm()
      |> reconcile_unrealized_casts()
      |> Beaver.Composer.run!()

    LLVMIR.translate!(llvm)
  end

  defp has_ex_ops?(module) do
    module
    |> MLIR.Module.body()
    |> Beaver.Walker.prewalk(false, fn
      %MLIR.Operation{} = op, acc ->
        {op, acc or String.starts_with?(MLIR.Operation.name(op), "ex.")}

      other, acc ->
        {other, acc}
    end)
    |> elem(1)
  end

  defp compile!(llvm_ir, target, entry, opt, nostdlib) do
    ll_path = tmp_path(".ll")
    wasm_path = tmp_path(".wasm")

    File.write!(ll_path, llvm_ir)

    args =
      ["cc", "-target", target, opt] ++
        if(nostdlib, do: ["-nostdlib"], else: []) ++
        ["-Wl,--no-entry", "-Wl,--export=#{entry}", "-o", wasm_path, ll_path]

    run_tool!(zig(), args)

    bytes = File.read!(wasm_path)
    cleanup([ll_path, wasm_path])
    bytes
  end

  defp zig do
    System.find_executable("zig") || raise "zig not found on PATH"
  end

  defp run_tool!(executable, args) do
    {output, status} = System.cmd(executable, args, stderr_to_stdout: true)

    unless status == 0 do
      raise "#{executable} failed (#{status}): #{String.slice(output, -1000, 1000)}"
    end

    :ok
  end

  defp tmp_path(suffix) do
    Path.join(
      System.tmp_dir!(),
      "beaver_wasm_#{System.unique_integer([:positive])}#{suffix}"
    )
  end

  defp cleanup(paths) do
    Enum.each(paths, &File.rm/1)
  end
end
