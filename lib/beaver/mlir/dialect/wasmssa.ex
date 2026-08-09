defmodule Beaver.MLIR.Dialect.WasmSSA do
  alias Beaver.MLIR.Dialect

  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.

  The WebAssembly SSA dialect is MLIR's native WASM IR (block/loop/if, locals,
  imports, tables, memories). Op construction functions are generated from the
  ops registered in the MLIR context; operand/attribute segment sizes and
  documentation come from the ODS dump.
  """

  use Beaver.MLIR.Dialect,
    dialect: "wasmssa",
    ops: Dialect.Registry.ops("wasmssa")

  @doc """
  Syntax sugar for `WasmSSA.func` SSA expression, mirroring `Func.func`.
  """
  defmacro func(call, do: body) do
    {func_name, args} = call |> Macro.decompose_call()

    quote do
      unquote(args)
      |> List.wrap()
      |> List.flatten()
      |> Keyword.put_new(
        Beaver.MLIR.SymbolTable.attribute_name(),
        Beaver.MLIR.Attribute.string(unquote(func_name))
      )
      |> Keyword.put_new(:loc, Beaver.MLIR.Location.from_env(__ENV__))
      |> then(
        &Beaver.MLIR.Operation.create(%Beaver.SSA{
          op: "wasmssa.func",
          ip: Beaver.Env.block(),
          ctx: Beaver.Env.context(),
          arguments: [fn -> unquote(body) end | &1]
        })
      )
    end
  end
end
