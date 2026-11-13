defmodule Beaver.MLIR.IRBatch.Protocol do
  @moduledoc """
  The batch wire layout shared with Zig. The JSON schema is consumed by Elixir
  at compilation and by `build.zig`; native field access and operand dispatch
  specialize from it at comptime. Version 1 accepts both single-block (7 fields)
  and CFG (8 fields) operations. Handle, bounds and context validation stays native.
  """
  @external_resource Path.join(__DIR__, "protocol.json")
  @schema @external_resource |> File.read!() |> JSON.decode!()
  @fields Enum.map(@schema["operation_fields"], &String.to_atom/1)
  def schema, do: @schema

  def operation(fields) do
    values = Enum.map(@fields, &Map.fetch!(fields, &1))
    values = if is_nil(fields.block), do: Enum.take(values, @schema["operation_required"]), else: values
    List.to_tuple(values)
  end

  for {kind, tag} <- Enum.with_index(@schema["operand_kinds"]) do
    def operand(unquote(String.to_atom(kind)), index, value), do: {unquote(tag), index, value}
  end

  @doc "Returns the schema embedded in the loaded NIF, useful for diagnosing stale native builds."
  def native_schema(ctx), do: Beaver.Execution.run(ctx, :batch_schema, {}, strategy: :auto) |> JSON.decode!()
end
