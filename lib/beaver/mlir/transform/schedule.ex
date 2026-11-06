alias Beaver.MLIR.{Attribute, Context, Operation}

defmodule Beaver.MLIR.Transform.Schedule do
  @moduledoc false
  alias Beaver.MLIR
  alias Beaver.MLIR.Transform

  @doc false
  def with_module(input, %Context{} = ctx, fun) when is_function(fun, 1) do
    with {:ok, bytes} <- source_bytes(input),
         {:ok, module} <- parse_module(bytes, ctx) do
      try do
        case MLIR.verify(module) do
          {:ok, _module} -> fun.(module)
          :null -> {:error, invalid_error("parsed transform module is null")}
          {:error, diagnostics} -> {:error, invalid_error(:verification_failed, diagnostics)}
        end
      after
        MLIR.Module.destroy(module)
      end
    end
  rescue
    exception in [ArgumentError] -> {:error, invalid_error(Exception.message(exception))}
  end

  @doc false
  def find_sequence(%MLIR.Module{} = module, sequence) when is_binary(sequence) do
    module
    |> operations()
    |> Enum.find(fn operation ->
      Operation.name(operation) == "transform.named_sequence" and attribute_value(operation, "sym_name") == sequence
    end)
    |> case do
      nil -> {:error, invalid_error({:sequence_not_found, sequence})}
      operation -> {:ok, operation}
    end
  end

  defp source_bytes({format, bytes}) when format in [:text, :bytecode] and is_binary(bytes), do: {:ok, bytes}

  defp source_bytes(bytes) when is_binary(bytes), do: {:ok, bytes}
  defp source_bytes(%MLIR.Module{} = module), do: {:ok, MLIR.Bytecode.write!(module)}

  defp source_bytes(%Operation{} = op) do
    case enclosing_module(op) do
      nil ->
        text =
          "module attributes {transform.with_named_sequence} {\n" <>
            MLIR.to_string(op, generic: false) <> "\n}"

        {:ok, text}

      module_operation ->
        {:ok, MLIR.Bytecode.write!(module_operation)}
    end
  end

  defp source_bytes(input), do: {:error, invalid_error({:unsupported_schedule_input, input})}

  defp parse_module(bytes, ctx) do
    case MLIR.Module.create(bytes, ctx: ctx) do
      {:ok, module} -> {:ok, module}
      {:error, diagnostics} -> {:error, invalid_error(:parse_failed, diagnostics)}
    end
  end

  defp enclosing_module(op),
    do: if(Operation.name(op) == "builtin.module", do: op, else: enclosing_module_parent(Operation.parent(op)))

  defp enclosing_module_parent(op) do
    cond do
      MLIR.null?(op) -> nil
      Operation.name(op) == "builtin.module" -> op
      true -> enclosing_module_parent(Operation.parent(op))
    end
  end

  defp attribute_value(op, name) do
    case Operation.fetch(op, name) do
      {:ok, attribute} -> Attribute.value(attribute)
      :error -> nil
    end
  end

  defp operations(module) do
    {_module, operations} =
      Beaver.Walker.prewalk(module, [], fn
        %Operation{} = operation, acc -> {operation, [operation | acc]}
        entity, acc -> {entity, acc}
      end)

    Enum.reverse(operations)
  end

  defp invalid_error(reason, diagnostics \\ []),
    do: %Transform.Error{kind: :invalid_schedule, reason: reason, diagnostics: MLIR.Diagnostic.process(diagnostics)}
end
