defmodule BEAMSSATest do
  use ExUnit.Case

  test "parse ssa" do
    alias Beaver.BEAM.SSA

    SSA.from_file!("test/beam_ssa_pre_codegen/play_with_ssa.ex.erl")
    # |> IO.inspect(limit: :infinity)
    |> SSA.MLIRGen.module()
  end
end
