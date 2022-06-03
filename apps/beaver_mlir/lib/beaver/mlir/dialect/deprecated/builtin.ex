defmodule Beaver.MLIR.Dialect.Builtin.Deprecated do
  @moduledoc false
  require Logger
  alias Beaver.MLIR.CAPI.IR

  defmodule Type do
    defmodule Function do
      @moduledoc """
      Struct to wrap a function type and provide helper functions to interact with function type attr.
      """
      @enforce_keys [:func_type, :inputs, :results]
      defstruct [:func_type, :inputs, :results]

      @doc """
      create a function TypeAttr from a string and and extract types from it
      """
      def from_attr(ctx, attr) when is_binary(attr) do
        function_type = IR.mlirAttributeParseGet(ctx, IR.string_ref(attr))
        function_type = function_type |> IR.mlirTypeAttrGetValue()

        is_func_type =
          function_type
          |> IR.mlirTypeIsAFunction()
          |> Exotic.Value.extract()

        if is_func_type do
          num_inputs =
            function_type
            |> IR.mlirFunctionTypeGetNumInputs()
            |> Exotic.Value.extract()

          num_results =
            function_type
            |> IR.mlirFunctionTypeGetNumResults()
            |> Exotic.Value.extract()

          inputs =
            for i <- 0..(num_inputs - 1)//1 do
              function_type
              |> IR.mlirFunctionTypeGetInput(i)
            end

          results =
            for i <- 0..(num_results - 1)//1 do
              function_type
              |> IR.mlirFunctionTypeGetResult(i)
            end

          %__MODULE__{
            func_type: function_type,
            inputs: inputs,
            results: results
          }
        else
          {:error, "not a valid func type: #{function_type}"}
        end
      end
    end
  end
end
