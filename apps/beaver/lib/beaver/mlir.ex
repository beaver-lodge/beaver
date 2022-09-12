defmodule Beaver.MLIR do
  @moduledoc """
  Provide macros to insert MLIR context and IR element of structure. These macros are designed to mimic the behavior and aesthetics of __MODULE__/0, __CALLER__/0 in Elixir.
  Its distinguished form is to indicate this should not be expected to be a function or a macro works like a function.
  """
  require Logger
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI

  alias Beaver.MLIR.Value

  alias Beaver.MLIR.CAPI.{
    MlirOperation,
    MlirAttribute,
    MlirBlock,
    MlirAffineExpr,
    MlirAffineMap,
    MlirIntegerSet,
    MlirType
  }

  def dump(%__MODULE__.Module{} = mlir) do
    CAPI.mlirModuleGetOperation(mlir)
    |> dump
  end

  def dump(%MlirOperation{} = mlir) do
    CAPI.mlirOperationDump(mlir)
    :ok
  end

  def dump(%MlirAttribute{} = mlir) do
    CAPI.mlirAttributeDump(mlir)
    :ok
  end

  def dump(%Value{} = mlir) do
    CAPI.mlirValueDump(mlir)
    :ok
  end

  def dump(%MlirAffineExpr{} = mlir) do
    CAPI.mlirAffineExprDump(mlir)
    :ok
  end

  def dump(%MlirAffineMap{} = mlir) do
    CAPI.mlirAffineMapDump(mlir)
    :ok
  end

  def dump(%MlirIntegerSet{} = mlir) do
    CAPI.mlirIntegerSetDump(mlir)
    :ok
  end

  def dump(%MlirType{} = mlir) do
    CAPI.mlirTypeDump(mlir)
    :ok
  end

  def dump(_) do
    :error
  end

  def dump!(mlir) do
    with :ok <- dump(mlir) do
      mlir
    else
      :error ->
        error_msg = "can't dump #{inspect(mlir)}"
        raise error_msg
    end
  end

  def is_null(%MlirAttribute{} = v) do
    CAPI.beaverAttributeIsNull(v) |> Beaver.Native.to_term()
  end

  def is_null(%MlirOperation{} = v) do
    CAPI.beaverOperationIsNull(v) |> Beaver.Native.to_term()
  end

  def is_null(%MlirBlock{} = v) do
    CAPI.beaverBlockIsNull(v) |> Beaver.Native.to_term()
  end

  def is_null(%Value{} = v) do
    CAPI.beaverValueIsNull(v) |> Beaver.Native.to_term()
  end

  def to_string(%MlirAttribute{ref: ref}) do
    CAPI.beaver_raw_beaver_attribute_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  def to_string(%Value{ref: ref}) do
    CAPI.beaver_raw_beaver_value_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  def to_string(%MlirOperation{ref: ref}) do
    CAPI.beaver_raw_beaver_operation_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  def to_string(%__MODULE__.Module{} = module) do
    module |> __MODULE__.Operation.from_module() |> __MODULE__.to_string()
  end

  def to_string(%MlirType{ref: ref}) do
    CAPI.beaver_raw_beaver_type_to_charlist(ref) |> Beaver.Native.check!() |> List.to_string()
  end

  def to_string(%MlirAffineMap{ref: ref}) do
    CAPI.beaver_raw_beaver_affine_map_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  defmacro __CONTEXT__() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_internal_env_ctx, nil}) do
      quote do
        Kernel.var!(beaver_internal_env_ctx)
      end
    else
      raise "no MLIR context in environment, maybe you forgot to put the ssa form inside the 'mlir ctx: ctx, do: ....' ?"
    end
  end

  defmacro __MODULE__() do
    quote do
      raise "TODO: impl me"
    end
  end

  defmacro __PARENT__() do
    quote do
      raise "TODO: impl me"
    end
  end

  defmacro __REGION__() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_env_region, nil}) do
      quote do
        Kernel.var!(beaver_env_region)
      end
    else
      quote do
        nil
      end
    end
  end

  defmacro __BLOCK__() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_internal_env_block, nil}) do
      quote do
        Kernel.var!(beaver_internal_env_block)
      end
    else
      raise "no block in environment, maybe you forgot to put the ssa form inside the Beaver.mlir/2 macro or a block/1 macro?"
    end
  end

  defmacro __BLOCK__({var_name, _line, nil} = block_var) do
    if Macro.Env.has_var?(__CALLER__, {var_name, nil}) do
      quote do
        %Beaver.MLIR.CAPI.MlirBlock{} = unquote(block_var)
      end
    else
      quote do
        Kernel.var!(unquote(block_var)) = Beaver.MLIR.Block.create([])
        %Beaver.MLIR.CAPI.MlirBlock{} = Kernel.var!(unquote(block_var))
      end
    end
  end

  defmacro __LOCATION__() do
    raise "TODO: create location from Elixir caller"
  end
end
