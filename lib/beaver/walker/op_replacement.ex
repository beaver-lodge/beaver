defmodule Beaver.Walker.OpReplacement do
  @moduledoc """
  A placeholder when an operation is replaced by value.
  """
  @type t() :: %__MODULE__{
          operands: Beaver.Walker.t() | list(),
          attributes: Beaver.Walker.t() | list(),
          results: Beaver.Walker.t() | list(),
          successors: Beaver.Walker.t() | list(),
          regions: Beaver.Walker.t() | list()
        }
  defstruct operands: [], attributes: [], results: [], successors: [], regions: []
end
