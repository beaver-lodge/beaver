defmodule Beaver.MLIR.Trait do
  @moduledoc """
  Attaches and queries traits on dynamically defined MLIR operations.

  Trait registration belongs to an `MLIR.Context`. Attaching an already
  present trait is a no-op, so a dynamic dialect's runtime extensions may be
  installed more than once in the same context.

  The target operation must already be registered by an extensible dialect.
  """

  alias Beaver.MLIR

  @traits %{
    terminator: {
      :mlirDynamicOpTraitIsTerminatorCreate,
      :mlirDynamicOpTraitIsTerminatorGetTypeID
    },
    isolated_from_above: {
      :mlirDynamicOpTraitIsIsolatedFromAboveCreate,
      :mlirDynamicOpTraitIsIsolatedFromAboveGetTypeID
    },
    no_terminator: {
      :mlirDynamicOpTraitNoTerminatorCreate,
      :mlirDynamicOpTraitNoTerminatorGetTypeID
    }
  }

  @type builtin :: :terminator | :isolated_from_above | :no_terminator

  @doc "Returns the built-in dynamic traits supported by MLIR."
  @spec builtins() :: [builtin()]
  def builtins, do: Map.keys(@traits)

  @doc false
  def normalize!(nil), do: []

  def normalize!(traits) when is_list(traits) do
    traits = Enum.uniq(traits)
    unsupported = traits -- builtins()

    if unsupported != [] do
      raise ArgumentError, "unsupported Slang traits: #{inspect(unsupported)}"
    end

    if :terminator in traits and :no_terminator in traits do
      raise ArgumentError,
            "conflicting Slang traits: :terminator and :no_terminator cannot be combined"
    end

    traits
  end

  def normalize!(traits) do
    raise ArgumentError, "expected a list of Slang traits, got: #{inspect(traits)}"
  end

  @doc "Attaches a built-in trait to a registered dynamic operation."
  @spec attach(MLIR.Context.t(), String.t(), builtin()) :: :ok
  def attach(%MLIR.Context{} = context, operation_name, trait)
      when is_binary(operation_name) do
    if has?(context, operation_name, trait) do
      :ok
    else
      {create, _type_id} = definition!(trait)
      dynamic_trait = apply(MLIR.CAPI, create, [])
      operation_name_ref = MLIR.StringRef.create(operation_name)

      attached? =
        MLIR.CAPI.mlirDynamicOpTraitAttach(dynamic_trait, operation_name_ref, context)
        |> Beaver.Native.to_term()

      # mlirDynamicOpTraitAttach consumes the trait on both success and
      # failure. A false result can be an idempotent concurrent attachment, so
      # inspect the registered operation again without destroying the pointer.
      if attached? or has?(context, operation_name_ref, trait) do
        :ok
      else
        raise ArgumentError,
              "failed to attach #{inspect(trait)} to dynamic operation #{operation_name}"
      end
    end
  end

  @doc "Attaches all declared traits for operations in a dynamic dialect."
  @spec attach_all(MLIR.Context.t(), String.t(), [{String.t(), [builtin()]}]) :: :ok
  def attach_all(%MLIR.Context{} = context, dialect, declarations)
      when is_binary(dialect) and is_list(declarations) do
    for {operation, traits} <- declarations,
        trait <- traits do
      attach(context, "#{dialect}.#{operation}", trait)
    end

    :ok
  end

  @doc "Checks whether a registered operation name has a built-in trait."
  @spec has?(MLIR.Context.t(), String.t() | MLIR.StringRef.t(), builtin()) :: boolean()
  def has?(%MLIR.Context{} = context, operation_name, trait) do
    operation_name =
      case operation_name do
        %MLIR.StringRef{} -> operation_name
        name when is_binary(name) -> MLIR.StringRef.create(name)
      end

    MLIR.CAPI.mlirOperationNameHasTrait(operation_name, type_id(trait), context)
    |> Beaver.Native.to_term()
  end

  @doc "Checks whether an operation has a built-in trait."
  @spec has?(MLIR.Operation.t(), builtin()) :: boolean()
  def has?(%MLIR.Operation{} = operation, trait) do
    has?(MLIR.context(operation), MLIR.Operation.name(operation), trait)
  end

  @doc false
  def type_id(trait) do
    {_create, type_id} = definition!(trait)
    apply(MLIR.CAPI, type_id, [])
  end

  defp definition!(trait) do
    case @traits do
      %{^trait => definition} -> definition
      _ -> raise ArgumentError, "unsupported MLIR trait: #{inspect(trait)}"
    end
  end
end
