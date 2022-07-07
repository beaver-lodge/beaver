defmodule Beaver.DSL.Op.Prototype do
  @moduledoc """
  Beaver.DSL.Prototype is a struct holding an Op's operands, results, attributes before creating it. It is similar to Beaver.DSL.SSA but there are applicable in different scenarios. Beaver.DSL.Prototype should be used where it is desired to get the created MLIR Op rather than the MLIR values of its results.
  Unlike SSA, operands/attributes/results in Prototype don't necessary contain real MLIR values/attributes of a operation. They could be other types for different abstraction or representation.
  For instance, when Prototype is used to compiling Elixir patterns to PDL, these fields contains MLIR values of PDL handles.
  """

  defstruct operands: [], attributes: [], results: [], successors: [], regions: []

  defmacro __using__(opts) do
    op_name = Keyword.fetch!(opts, :op_name)

    quote do
      @behaviour Beaver.DSL.Op.Prototype
      defstruct operands: [], attributes: [], results: [], successors: [], regions: []

      require Logger
      @impl true
      def op_name() do
        unquote(op_name)
      end

      @on_load :register_op_prototype

      def register_op_prototype do
        Beaver.MLIR.DSL.Op.Registry.register(unquote(op_name), __MODULE__)
        :ok
      end
    end
  end

  @doc """
  Dispatch the op name and map to the callback `cb` if this is a module implement the behavior this module define.
  """
  def dispatch(module, fields, cb) when is_function(cb, 2) do
    dispatch(module, fields, :placeholder, fn n, p, :placeholder -> cb.(n, p) end)
  end

  def dispatch(module, fields, extra_arg, cb) when is_function(cb, 3) do
    if is_compliant(module) do
      cb.(module.op_name(), struct!(Beaver.DSL.Op.Prototype, fields), extra_arg)
    else
      struct!(module, fields)
    end
  end

  @doc """
  check if this module exist and compliant to Op.Prototype
  """
  def is_compliant(module) do
    case Code.ensure_loaded(module) do
      {:module, module} ->
        function_exported?(module, :__info__, 1) and
          __MODULE__ in (module.module_info[:attributes][:behaviour] || [])

      {:error, :nofile} ->
        false
    end
  end

  @callback op_name() :: String.t()

  def op_name!(op_module) do
    if is_compliant(op_module) do
      op_module.op_name()
    else
      raise "required a registered op like Beaver.MLIR.Dialect.Func.Func"
    end
  end
end
