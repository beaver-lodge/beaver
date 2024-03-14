defmodule Beaver.MLIR.Dialect.Builtin do
  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """
  use Beaver.MLIR.Dialect,
    dialect: "builtin",
    ops: Beaver.MLIR.Dialect.Registry.ops("builtin") |> Enum.reject(fn x -> x in ~w{module} end)

  @doc """
  Macro to create a module and insert ops into its body. region/1 shouldn't be called because region of one block will be created.
  """
  defmacro module(attrs \\ [], do: block) do
    {block_cache, block_restore} = Beaver.parent_scope_block_caching(__CALLER__)

    quote do
      ctx = Beaver.Env.context()

      location =
        Keyword.get(unquote(attrs), :loc) ||
          Beaver.MLIR.Location.file(
            name: __ENV__.file,
            line: __ENV__.line,
            ctx: ctx
          )

      module = Beaver.MLIR.CAPI.mlirModuleCreateEmpty(location)

      for {name, attr} <- unquote(attrs) do
        name = name |> Beaver.MLIR.StringRef.create()

        attr = Beaver.Deferred.create(attr, ctx)

        module
        |> Beaver.MLIR.CAPI.mlirModuleGetOperation()
        |> Beaver.MLIR.CAPI.mlirOperationSetAttributeByName(name, attr)
      end

      module_body_block = Beaver.MLIR.CAPI.mlirModuleGetBody(module)
      unquote(block_cache)
      Kernel.var!(beaver_internal_env_block) = module_body_block
      %Beaver.MLIR.Block{} = Kernel.var!(beaver_internal_env_block)
      unquote(block)
      unquote(block_restore)
      module
    end
  end
end
