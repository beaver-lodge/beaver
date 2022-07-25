defmodule Fizz.CodeGen.NIF do
  alias Fizz.CodeGen.{Function, Type}
  @type dirty() :: :io | :cpu | false
  @type t() :: %__MODULE__{
          name: String.t(),
          nif_name: String.t(),
          arity: integer(),
          ret: String.t(),
          dirty: dirty()
        }
  defstruct name: nil, nif_name: nil, arity: 0, ret: nil, dirty: false

  def from_function(%Fizz.CodeGen.Function{name: name, args: args, ret: ret}) do
    %__MODULE__{
      name: name,
      arity: length(args),
      ret: ret
    }
  end

  def gen(%__MODULE__{name: name, nif_name: nif_name, arity: arity, dirty: dirty}) do
    dirty_flag =
      case dirty do
        :cpu -> "1"
        :io -> "2"
        false -> "0"
      end

    """
    e.ErlNifFunc{.name = "#{nif_name}", .arity = #{arity}, .fptr = #{name}, .flags = #{dirty_flag}},
    """
  end
end
