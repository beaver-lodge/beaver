defmodule Beaver.MLIR.Dialect.Ex do
  @moduledoc """
  Elixir IR composed from language, execution and runtime schemas.
  Existing `ex.*` builders and types remain available through this facade.
  See [layer contracts](dialect-conversion.html#ex-provider-boundary).
  """
  alias Beaver.MLIR.Operation

  @schema_files [__ENV__.file | Enum.map(~w(language execution runtime schema), &Path.join(__DIR__, "ex/#{&1}.ex"))]
  for path <- @schema_files, do: @external_resource(path)

  @schema_digest "sha256:" <>
                   Base.encode16(:crypto.hash(:sha256, Enum.map(@schema_files, &File.read!/1)), case: :lower)

  use Beaver.Slang, name: "ex"
  import Beaver.Slang, except: [defop: 1, defop: 2]
  import Beaver.MLIR.Dialect.Ex.Schema, only: [defop: 2, defops: 3]
  Module.register_attribute(__MODULE__, :ex_operations, accumulate: true)

  deftype term()
  deftype bound()
  deftype unbound()

  defconstraint integer_value, do: base("#builtin.integer")

  defconstraint callee_value, do: base("#builtin.string")

  defconstraint cmp_predicate_value, do: base("#builtin.string")

  # Named constraints avoid repeating MLIR builder ASTs.
  defconstraint any_value, do: any()

  defconstraint integer_like, do: all_of([base("!builtin.integer"), any()])

  defconstraint term_like, do: any_of([base(term()), base(bound()), base(unbound())])

  use Beaver.MLIR.Dialect.Ex.Language
  use Beaver.MLIR.Dialect.Ex.Execution
  use Beaver.MLIR.Dialect.Ex.Runtime

  @operations Map.new(@ex_operations)
  if map_size(@operations) != length(@ex_operations), do: raise("duplicate Ex operation ownership")

  @doc "Content identity of the facade and every schema fragment, including interfaces."
  def schema_digest, do: @schema_digest

  @doc "The three owners of Ex operations."
  def layers, do: [:language, :execution, :runtime]

  @doc "Registered operation names, optionally restricted to one layer."
  def operations(layer \\ :all)
  def operations(:all), do: @operations |> Map.keys() |> Enum.sort()

  def operations(layer) when layer in [:language, :execution, :runtime],
    do: for({name, ^layer} <- @operations, do: name) |> Enum.sort()

  @doc "Returns the owner of a registered, fully qualified Ex operation. Unknown names reject."
  def layer(name), do: Map.fetch!(@operations, name)

  @doc "Verifies MLIR and rejects Ex operations outside the explicitly allowed layers."
  def verify_layers!(ir, allowed) do
    allowed = List.wrap(allowed)

    unless allowed != [] and allowed -- layers() == [],
      do: raise(ArgumentError, "invalid Ex layers: #{inspect(allowed)}")

    Beaver.MLIR.verify!(ir)

    Beaver.Walker.prewalk(ir, fn
      %Operation{} = op ->
        name = Operation.name(op)

        if String.starts_with?(name, "ex.") and layer(name) not in allowed,
          do: raise(ArgumentError, "#{name} belongs to #{layer(name)}, allowed: #{inspect(allowed)}")

        op

      element ->
        element
    end)

    ir
  end
end
