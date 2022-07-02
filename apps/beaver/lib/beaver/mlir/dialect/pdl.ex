defmodule Beaver.MLIR.Dialect.PDL do
  use Beaver.MLIR.Dialect.Generator,
    dialect: "pdl",
    ops: Beaver.MLIR.Dialect.Registry.ops("pdl")

  @moduledoc """
  This module provides functions to compile definitions in Elixir to PDL IR. Usually the generated PDL patterns is used in a MLIR external pass in Elixir. With proper composition, functions in this module should provide equivalent features of PDLL.
  """

  defmodule Constraint do
    @moduledoc """
    PDL constraints are kind of like guard functions/macros in Elixir. All pattern matching in the arguments of a Elixir defintion will be compiled to PDL constraints.
    """
  end

  defmodule Rewrite do
    defstruct name: nil
  end

  defmodule Pattern do
    defstruct constraints: [], rewrites: []
  end

  defmodule Operation do
    defmodule State do
      alias Beaver.MLIR.Operation.State

      def get(location: location, name: name) do
        get(location: location, name: name, operands: [])
      end

      def get(location: location, name: name, operands: operands) do
        State.get!("pdl.operation", location)
        |> State.add_attr(
          attributeNames: "[]",
          name: "\"#{name}\"",
          # TODO: update elements by length of operands
          operand_segment_sizes: "dense<0> : vector<3xi32>"
        )
        |> State.add_operand(operands)
        |> State.add_result(["!pdl.operation"])
      end
    end
  end

  defmodule Erase do
    defmodule State do
      alias Beaver.MLIR.Operation.State

      def get(location: location, erasee: erasee) do
        State.get!("pdl.erase", location)
        |> State.add_operand([erasee])
      end
    end
  end
end
