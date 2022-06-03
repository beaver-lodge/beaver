defmodule Exotic.Function.Definition do
  @enforce_keys [:name, :return_type, :arg_types]
  defstruct [:name, :return_type, :arg_types]

  def from_code_gen(%Exotic.CodeGen.Function{name: name, args: args, ret: ret}) do
    args =
      for {_name, type} <- args do
        type
      end

    %Exotic.Function.Definition{
      name: name,
      return_type: ret,
      arg_types: args
    }
  end
end

defmodule Exotic.Function do
  @enforce_keys [:ref, :def]
  defstruct [:ref, :def]

  def get(
        %__MODULE__.Definition{name: name, return_type: return_type, arg_types: arg_types} = def
      ) do
    return_type = return_type |> Exotic.Type.get() |> Map.get(:ref)

    arg_types =
      arg_types
      |> Enum.map(&Exotic.Type.get/1)
      |> Enum.map(&Map.get(&1, :ref))

    %__MODULE__{ref: Exotic.NIF.get_func(name, return_type, arg_types), def: def}
  end
end
