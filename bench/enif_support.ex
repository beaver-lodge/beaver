defmodule ENIFSupport do
  @moduledoc false
  defstruct [:mod, :engine]
  alias Beaver.MLIR
  import MLIR.Conversion

  @callback after_verification(any()) :: any()
  @callback create(ctx :: MLIR.Context.t()) :: any()

  defmacro __using__(_opts) do
    quote do
      def after_verification(op), do: op
      defoverridable after_verification: 1

      @behaviour ENIFSupport
      def init(ctx) do
        create(ctx)
        |> MLIR.verify!()
        |> after_verification()
        |> ENIFSupport.lower()
      end
    end
  end

  @print_flag "ENIF_SUPPORT_PRINT_IR"
  def lower(op) do
    op
    |> then(&if(System.get_env(@print_flag), do: MLIR.Transform.print_ir(&1), else: &1))
    |> Beaver.Composer.nested("func.func", "llvm-request-c-wrappers")
    |> convert_scf_to_cf
    |> convert_arith_to_llvm()
    |> convert_index_to_llvm()
    |> convert_func_to_llvm()
    |> Beaver.Composer.append("convert-vector-to-llvm{reassociate-fp-reductions}")
    |> Beaver.Composer.append("finalize-memref-to-llvm")
    |> then(&if(System.get_env(@print_flag), do: MLIR.Transform.print_ir(&1), else: &1))
    |> reconcile_unrealized_casts
    |> Beaver.Composer.run!()
    |> then(fn m ->
      e = MLIR.ExecutionEngine.create!(m, opt_level: 3) |> Beaver.ENIF.register_symbols()
      %ENIFSupport{mod: m, engine: e}
    end)
  end

  def destroy(%ENIFSupport{mod: m, engine: e}) do
    MLIR.ExecutionEngine.destroy(e)
    MLIR.Module.destroy(m)
  end
end
