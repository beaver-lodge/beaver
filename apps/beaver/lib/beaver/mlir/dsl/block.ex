defmodule Beaver.MLIR.DSL.Block do
  @moduledoc false
  # Transform the ast of a elixir call into a block creation and block args bindings
  def transform_call(call) do
    {bb_name, args} = call |> Macro.decompose_call()
    if not is_atom(bb_name), do: raise("block name must be an atom")

    opts = List.last(args)
    # transform {arg, type} into arg = type
    args_type_ast =
      for {var, type} <- args do
        quote do
          unquote(var) = unquote(type)
        end
      end

    # to be spliced into a list, and then as argument for Beaver.MLIR.Block.create/2
    args_var_ast =
      for {var, _type} <- args do
        quote do
          unquote(var)
        end
      end

    # to be spliced into a list, and then as argument for Beaver.MLIR.Block.create/2
    # TODO: use arg's location
    locations_var_ast =
      for _ <- args do
        quote do
          Beaver.MLIR.Managed.Location.get()
        end
      end

    # generate `var = mlir block arg` bindings for the uses in the do block
    block_arg_var_ast =
      for {{var, _}, index} <- Enum.with_index(args) do
        quote do
          unquote(var) =
            Beaver.MLIR.Managed.Block.get()
            |> Beaver.MLIR.Block.get_arg!(unquote(index))
        end
      end

    {
      args,
      opts,
      args_type_ast,
      args_var_ast,
      locations_var_ast,
      block_arg_var_ast
    }
  end
end
