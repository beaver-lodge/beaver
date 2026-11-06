alias Beaver.MLIR.{CAPI, Context, Operation, StringRef}
alias Beaver.Native

defmodule Beaver.MLIR.Trait do
  @moduledoc "Attaches and queries the built-in traits of dynamic MLIR operations."
  alias Beaver.MLIR

  @traits %{
    terminator: {:mlirDynamicOpTraitIsTerminatorCreate, :mlirDynamicOpTraitIsTerminatorGetTypeID},
    isolated_from_above:
      {:mlirDynamicOpTraitIsIsolatedFromAboveCreate, :mlirDynamicOpTraitIsIsolatedFromAboveGetTypeID},
    no_terminator: {:mlirDynamicOpTraitNoTerminatorCreate, :mlirDynamicOpTraitNoTerminatorGetTypeID}
  }

  @type builtin :: :terminator | :isolated_from_above | :no_terminator
  @type declaration :: builtin()

  @doc "Returns the built-in dynamic traits supported by MLIR."
  @spec builtins() :: [builtin()]
  def builtins, do: Map.keys(@traits)

  @doc false
  def normalize!(nil), do: []

  def normalize!(traits) when is_list(traits) do
    traits = Enum.uniq(traits)
    unsupported = Enum.reject(traits, &is_map_key(@traits, &1))

    if unsupported != [], do: raise(ArgumentError, "unsupported Slang traits: #{inspect(unsupported)}")

    if :terminator in traits and :no_terminator in traits,
      do: raise(ArgumentError, "conflicting Slang traits: :terminator and :no_terminator cannot be combined")

    traits
  end

  def normalize!(traits), do: raise(ArgumentError, "expected a list of Slang traits, got: #{inspect(traits)}")

  @doc "Attaches a trait to a registered dynamic operation."
  @spec attach(Context.t(), String.t(), declaration()) :: :ok
  def attach(%Context{} = ctx, op_name, trait)
      when is_binary(op_name) and is_atom(trait) do
    if has?(ctx, op_name, trait) do
      :ok
    else
      {create, _type_id} = definition!(trait)
      dynamic_trait = apply(CAPI, create, [])
      operation_name_ref = StringRef.create(op_name)

      attached? = CAPI.mlirDynamicOpTraitAttach(dynamic_trait, operation_name_ref, ctx) |> Native.to_term()

      # mlirDynamicOpTraitAttach consumes the trait on both success and
      # failure. A false result can be an idempotent concurrent attachment, so
      # inspect the registered operation again without destroying the pointer.
      if attached? or has?(ctx, operation_name_ref, trait),
        do: :ok,
        else: raise(ArgumentError, "failed to attach #{inspect(trait)} to dynamic operation #{op_name}")
    end
  end

  @doc "Attaches all declared traits for operations in a dynamic dialect."
  @spec attach_all(Context.t(), String.t(), [{String.t(), [declaration()]}]) :: :ok
  def attach_all(%Context{} = ctx, dialect, declarations)
      when is_binary(dialect) and is_list(declarations) do
    for {operation, traits} <- declarations, trait <- traits do
      attach(ctx, "#{dialect}.#{operation}", trait)
    end

    :ok
  end

  @doc "Checks whether a registered operation name has a trait."
  @spec has?(Context.t(), String.t() | StringRef.t(), builtin()) ::
          boolean()
  def has?(%Context{} = ctx, op_name, trait),
    do: has_type_id?(ctx, op_name, type_id(trait))

  @doc "Checks whether an operation has a trait."
  @spec has?(Operation.t(), builtin()) :: boolean()
  def has?(%Operation{} = op, trait),
    do: has?(MLIR.context(op), Operation.name(op), trait)

  @doc false
  def type_id(trait) do
    {_create, type_id} = definition!(trait)
    apply(CAPI, type_id, [])
  end

  defp has_type_id?(ctx, op_name, type_id) do
    op_name =
      case op_name do
        %StringRef{} -> op_name
        name when is_binary(name) -> StringRef.create(name)
      end

    CAPI.mlirOperationNameHasTrait(op_name, type_id, ctx) |> Native.to_term()
  end

  defp definition!(trait) do
    case @traits do
      %{^trait => definition} -> definition
      _ -> raise ArgumentError, "unsupported MLIR trait: #{inspect(trait)}"
    end
  end
end
