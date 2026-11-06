alias Beaver.MLIR.{Context, Operation, StringRef, Value}
alias Beaver.Native

defmodule Beaver.MLIR.Location do
  @moduledoc "Source locations with deferred construction and context-checked resolution."
  alias Beaver.Deferred
  alias Beaver.MLIR.CAPI
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native

  @type source() :: t() | Operation.t() | Value.t() | Deferred.contextual(t() | Operation.t() | Value.t())

  def file(opts) do
    name = StringRef.create(Keyword.fetch!(opts, :name))
    line = Keyword.fetch!(opts, :line)
    column = opts[:column] || 0
    Deferred.from_opts(opts, &CAPI.mlirLocationFileLineColGet(&1, name, line, column))
  end

  def from_env(env, opts \\ []) when is_map(env) do
    line =
      case Map.get(env, :line) do
        line when is_integer(line) -> line
        _ -> 0
      end

    file(opts |> Keyword.put(:name, to_string(Map.get(env, :file, "nofile"))) |> Keyword.put(:line, line))
  end

  def unknown(opts \\ []), do: Deferred.from_opts(opts, &CAPI.mlirLocationUnknownGet/1)

  def of(%__MODULE__{} = loc), do: loc
  def of(%Operation{} = op), do: Operation.location(op)
  def of(%Value{} = value), do: Value.location(value)

  def of(%Deferred{} = deferred), do: Deferred.defer(fn ctx -> deferred |> Deferred.resolve(ctx) |> of() end)

  def of(other) do
    raise ArgumentError, "expected an MLIR location, operation, value, or deferred equivalent, got: #{inspect(other)}"
  end

  @doc "Resolves a location-bearing value in `ctx` and verifies context ownership."
  @spec resolve(source(), Context.t()) :: t()
  def resolve(source, %Context{} = ctx) do
    location = source |> Deferred.resolve(ctx) |> of()

    Context.ensure_same!(location, ctx)
  end
end
