defmodule Beaver.MLIR.Trait do
  @moduledoc """
  Attaches and queries traits on dynamically defined MLIR operations.

  Trait registration belongs to an `MLIR.Context`. Attaching an already
  present trait is a no-op, so a dynamic dialect's runtime extensions may be
  installed more than once in the same context.

  Custom traits use a stable TypeID per identity and context. Their verifier
  callbacks run in a dedicated BEAM process and receive a borrowed operation
  that is valid only until the callback returns. The target operation must
  already be registered by an extensible dialect.
  """

  use GenServer

  require Logger

  alias Beaver.MLIR
  alias Kinda.CallbackRuntime

  defmodule Attachment do
    @moduledoc "A live callback-backed dynamic trait attachment."
    @enforce_keys [:pid, :id, :identity, :operation_name, :type_id]
    defstruct [:pid, :id, :identity, :operation_name, :type_id]

    @type t :: %__MODULE__{
            pid: pid(),
            id: term(),
            identity: atom(),
            operation_name: String.t(),
            type_id: Beaver.MLIR.TypeID.t()
          }
  end

  defmodule DefinitionServer do
    @moduledoc false
    use GenServer

    alias Beaver.MLIR

    @impl true
    def init({context_ref, context_registry}) do
      {:ok, _} = Registry.register(context_registry, context_ref, :definition)
      {:ok, %{allocator: MLIR.CAPI.mlirTypeIDAllocatorCreate(), type_ids: %{}}}
    end

    @impl true
    def handle_call({:type_id, identity}, _from, state) do
      {type_id, type_ids} =
        Map.get_and_update(state.type_ids, identity, fn
          nil ->
            type_id = MLIR.CAPI.mlirTypeIDAllocatorAllocateTypeID(state.allocator)
            {type_id, type_id}

          type_id ->
            {type_id, type_id}
        end)

      {:reply, type_id, %{state | type_ids: type_ids}}
    end

    @impl true
    def terminate(_reason, state) do
      MLIR.CAPI.mlirTypeIDAllocatorDestroy(state.allocator)
      :ok
    end
  end

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

  @definition_registry __MODULE__.DefinitionRegistry
  @context_registry __MODULE__.ContextRegistry
  @custom_callback_keys [:verify, :verify_regions]

  @type builtin :: :terminator | :isolated_from_above | :no_terminator
  @type identity :: module() | atom()
  @type custom :: {identity(), keyword()}
  @type declaration :: builtin() | custom()

  @doc false
  def global_registrar_child_specs do
    [
      {Registry, keys: :unique, name: @definition_registry},
      {Registry, keys: :duplicate, name: @context_registry}
    ]
  end

  @doc false
  def release_context(%MLIR.Context{ref: context_ref}) do
    if Process.whereis(@context_registry) do
      entries = Registry.lookup(@context_registry, context_ref)

      entries
      |> Enum.filter(&(elem(&1, 1) == :attachment))
      |> Enum.each(fn {pid, _kind} -> release_attachment(pid) end)

      entries
      |> Enum.filter(&(elem(&1, 1) == :definition))
      |> Enum.each(fn {pid, _kind} -> stop_definition(pid) end)
    end

    :ok
  end

  @doc "Returns the built-in dynamic traits supported by MLIR."
  @spec builtins() :: [builtin()]
  def builtins, do: Map.keys(@traits)

  @doc false
  def normalize!(nil), do: []

  def normalize!(traits) when is_list(traits) do
    traits = Enum.uniq(traits)
    unsupported = Enum.reject(traits, &(builtin?(&1) or custom_declaration?(&1)))

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

  @doc "Attaches a trait to a registered dynamic operation."
  @spec attach(MLIR.Context.t(), String.t(), declaration()) :: :ok | Attachment.t()
  def attach(%MLIR.Context{} = context, operation_name, trait)
      when is_binary(operation_name) and is_atom(trait) do
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

  def attach(%MLIR.Context{} = context, operation_name, {identity, callbacks})
      when is_binary(operation_name) do
    attach_custom(context, operation_name, identity, callbacks)
  end

  @doc "Attaches a callback-backed custom trait to a dynamic operation."
  @spec attach_custom(MLIR.Context.t(), String.t(), identity(), keyword(), keyword()) ::
          :ok | Attachment.t()
  def attach_custom(context, operation_name, identity, callbacks, opts \\ []) do
    callbacks = normalize_callbacks!(callbacks)
    timeout_ms = Keyword.get(opts, :timeout, 30_000)

    unless is_integer(timeout_ms) and timeout_ms >= 0 do
      raise ArgumentError, ":timeout must be a non-negative integer"
    end

    type_id = type_id(context, identity)

    if has_type_id?(context, operation_name, type_id) do
      :ok
    else
      init = {context, operation_name, identity, type_id, callbacks, timeout_ms}

      case GenServer.start(__MODULE__, init) do
        {:ok, pid} ->
          GenServer.call(pid, :attachment)

        {:error, {%_{} = exception, stacktrace}} ->
          reraise exception, stacktrace

        {:error, reason} ->
          raise "failed to attach trait #{inspect(identity)}: #{inspect(reason)}"
      end
    end
  end

  @doc "Attaches all declared traits for operations in a dynamic dialect."
  @spec attach_all(MLIR.Context.t(), String.t(), [{String.t(), [declaration()]}]) :: :ok
  def attach_all(%MLIR.Context{} = context, dialect, declarations)
      when is_binary(dialect) and is_list(declarations) do
    for {operation, traits} <- declarations,
        trait <- traits do
      attach(context, "#{dialect}.#{operation}", trait)
    end

    :ok
  end

  @doc "Checks whether a registered operation name has a trait."
  @spec has?(MLIR.Context.t(), String.t() | MLIR.StringRef.t(), builtin() | identity()) ::
          boolean()
  def has?(%MLIR.Context{} = context, operation_name, trait) do
    has_type_id?(context, operation_name, type_id(context, trait))
  end

  @doc "Checks whether an operation has a trait."
  @spec has?(MLIR.Operation.t(), builtin() | identity()) :: boolean()
  def has?(%MLIR.Operation{} = operation, trait) do
    has?(MLIR.context(operation), MLIR.Operation.name(operation), trait)
  end

  @doc "Returns the TypeID used by a trait in a context."
  def type_id(%MLIR.Context{} = context, trait) do
    if builtin?(trait) do
      type_id(trait)
    else
      context
      |> definition_server()
      |> GenServer.call({:type_id, trait})
    end
  end

  @doc false
  def type_id(trait) do
    {_create, type_id} = definition!(trait)
    apply(MLIR.CAPI, type_id, [])
  end

  @impl true
  def init({context, operation_name, identity, type_id, callbacks, timeout_ms}) do
    {:ok, _} = Registry.register(@context_registry, context.ref, :attachment)

    {id, native_owner} =
      MLIR.CAPI.beaver_raw_dynamic_trait_attach(
        context,
        operation_name,
        type_id,
        callbacks[:verify],
        callbacks[:verify_regions],
        timeout_ms
      )

    {:ok,
     %{
       context: context,
       operation_name: operation_name,
       identity: identity,
       type_id: type_id,
       callbacks: callbacks,
       id: id,
       native_owner: native_owner
     }}
  end

  @impl true
  def handle_call(:attachment, _from, state) do
    attachment = %Attachment{
      pid: self(),
      id: state.id,
      identity: state.identity,
      operation_name: state.operation_name,
      type_id: state.type_id
    }

    {:reply, attachment, state}
  end

  def handle_call(:context_destroyed, _from, state) do
    :ok = MLIR.CAPI.beaver_raw_external_interface_release(state.native_owner)
    {:stop, :normal, :ok, state}
  end

  @impl true
  def handle_info(:external_interface_released, state), do: {:stop, :normal, state}

  def handle_info(message, state) do
    case dispatch(message, state.identity) do
      {:handled, _failure} ->
        {:noreply, state}

      :unhandled ->
        Logger.warning("unexpected dynamic trait callback message: #{inspect(message)}")
        {:noreply, state}
    end
  end

  defp dispatch({callback_name, token, callback, _id, operation}, identity)
       when callback_name in [:verify, :verify_regions] do
    operation = Beaver.Native.check!(operation)

    outcome =
      CallbackRuntime.invoke(
        token,
        fn -> callback.(operation) |> normalize_verification_result() end,
        &MLIR.CAPI.beaver_raw_callback_reply/2,
        &diagnose(&1, operation, identity, callback_name)
      )

    {:handled, callback_failure(outcome)}
  end

  defp dispatch(_message, _identity), do: :unhandled

  defp normalize_verification_result(result) when result in [:ok, true], do: {:ok, :ok}
  defp normalize_verification_result({:ok, _value}), do: {:ok, :ok}
  defp normalize_verification_result(result) when result in [:error, false], do: {:error, result}
  defp normalize_verification_result({:error, _reason} = error), do: error

  defp normalize_verification_result(other) do
    raise ArgumentError,
          "trait verifier must return :ok, true, {:ok, value}, :error, false, " <>
            "or {:error, reason}, got: #{inspect(other)}"
  end

  defp diagnose({:exception, kind, reason, stacktrace}, operation, identity, callback_name) do
    message =
      "dynamic trait #{inspect(identity)} #{callback_name} callback raised:\n" <>
        Exception.format(kind, reason, stacktrace)

    MLIR.Operation.location(operation) |> MLIR.Diagnostic.emit(message)
    Logger.error(message)
  end

  defp diagnose({:error, reason}, operation, identity, callback_name) do
    message =
      "dynamic trait #{inspect(identity)} #{callback_name} callback failed: #{inspect(reason)}"

    MLIR.Operation.location(operation) |> MLIR.Diagnostic.emit(message)
    Logger.error(message)
  end

  defp diagnose(_outcome, _operation, _identity, _callback_name), do: :ok

  defp normalize_callbacks!(callbacks) when is_list(callbacks) do
    unless Keyword.keyword?(callbacks) do
      raise ArgumentError, "custom trait callbacks must be a keyword list"
    end

    callbacks = Keyword.validate!(callbacks, @custom_callback_keys)

    if Enum.all?(@custom_callback_keys, &is_nil(callbacks[&1])) do
      raise ArgumentError, "custom trait requires :verify or :verify_regions"
    end

    Enum.each(callbacks, fn {name, callback} ->
      unless is_function(callback, 1) do
        raise ArgumentError, "custom trait #{inspect(name)} callback must have arity 1"
      end
    end)

    callbacks
  end

  defp normalize_callbacks!(callbacks) do
    raise ArgumentError,
          "custom trait callbacks must be a keyword list, got: #{inspect(callbacks)}"
  end

  defp has_type_id?(context, operation_name, type_id) do
    operation_name =
      case operation_name do
        %MLIR.StringRef{} -> operation_name
        name when is_binary(name) -> MLIR.StringRef.create(name)
      end

    MLIR.CAPI.mlirOperationNameHasTrait(operation_name, type_id, context)
    |> Beaver.Native.to_term()
  end

  defp definition_server(%MLIR.Context{ref: context_ref}) do
    name = {:via, Registry, {@definition_registry, context_ref}}

    case GenServer.start(DefinitionServer, {context_ref, @context_registry}, name: name) do
      {:ok, pid} -> pid
      {:error, {:already_started, pid}} -> pid
      {:error, reason} -> raise "failed to start trait definition registry: #{inspect(reason)}"
    end
  end

  defp release_attachment(pid) do
    GenServer.call(pid, :context_destroyed)
  catch
    :exit, {:noproc, _} -> :ok
    :exit, {:normal, _} -> :ok
  end

  defp stop_definition(pid) do
    GenServer.stop(pid, :normal)
  catch
    :exit, {:noproc, _} -> :ok
  end

  defp callback_failure({:error, reason}), do: {:error, reason}

  defp callback_failure({:exception, _kind, _reason, _stacktrace} = exception),
    do: exception

  defp callback_failure(_outcome), do: nil

  defp builtin?(trait), do: is_map_key(@traits, trait)

  defp custom_declaration?({identity, callbacks}) when is_list(callbacks) do
    identity != nil and Keyword.keyword?(callbacks) and
      Keyword.keys(callbacks) -- @custom_callback_keys == [] and
      Enum.any?(@custom_callback_keys, &Keyword.has_key?(callbacks, &1))
  end

  defp custom_declaration?(_trait), do: false

  defp definition!(trait) do
    case @traits do
      %{^trait => definition} -> definition
      _ -> raise ArgumentError, "unsupported MLIR trait: #{inspect(trait)}"
    end
  end
end
