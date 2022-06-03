defmodule Beaver.MLIR do
  defmacro block(call, do: block) do
    {bb, args} = call |> Macro.decompose_call() |> IO.inspect()
    if not is_atom(bb), do: raise("block name must be an atom")

    args_type_ast =
      for {var, val} <- args do
        quote do
          unquote(var) = unquote(val)
        end
      end

    args_var_ast =
      for {var, _val} <- args do
        quote do
          unquote(var)
        end
      end

    block_arg_var_ast =
      for {{var, _}, index} <- Enum.with_index(args) |> IO.inspect() do
        quote do
          unquote(var) = Beaver.MLIR.Block.get_arg!(block, unquote(index))
        end
      end

    # TODO: use arg's location
    locations_var_ast =
      for _ <- args do
        quote do
          Beaver.MLIR.Managed.Location.get()
        end
      end

    # block |> IO.inspect()

    block_ast =
      quote do
        unquote_splicing(args_type_ast)
        region = Beaver.MLIR.CAPI.mlirRegionCreate()
        block_arg_types = [unquote_splicing(args_var_ast)]
        block_arg_locs = [unquote_splicing(locations_var_ast)]
        block = Beaver.MLIR.Block.create(block_arg_types, block_arg_locs)
        unquote_splicing(block_arg_var_ast)
      end

    block_ast |> Macro.to_string() |> IO.puts()
    block_ast
  end
end
