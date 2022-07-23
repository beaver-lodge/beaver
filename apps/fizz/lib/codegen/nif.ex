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
      nif_name: Function.nif_func_name(name),
      arity: length(args),
      ret: ret
    }
  end

  def array_maker(type) do
    %__MODULE__{
      nif_name: Fizz.CodeGen.Function.array_maker_name(type),
      arity: 1,
      ret: Type.array_type_name(type)
    }
  end

  def ptr_maker(type) do
    %__MODULE__{
      nif_name: Fizz.CodeGen.Function.ptr_maker_name(type),
      arity: 1,
      ret: Type.ptr_type_name(type)
    }
  end

  def primitive_maker(type) do
    %__MODULE__{
      nif_name: Fizz.CodeGen.Function.primitive_maker_name(type),
      arity: 1
    }
  end

  def resource_maker(type) do
    %__MODULE__{
      nif_name: Fizz.CodeGen.Function.resource_maker_name(type),
      arity: 1
    }
  end

  def gen(%__MODULE__{nif_name: nif_name, arity: arity, dirty: dirty}) do
    dirty_flag =
      case dirty do
        :cpu -> "1"
        :io -> "2"
        false -> "0"
      end

    """
    e.ErlNifFunc{.name = "#{nif_name}", .arity = #{arity}, .fptr = #{nif_name}, .flags = #{dirty_flag}},
    """
  end
end
