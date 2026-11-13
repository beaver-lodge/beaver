alias Beaver.MLIR.{CAPI, Context}

defmodule Beaver.MLIR.ExternalInterface do
  @moduledoc """
  Owns callback-backed MLIR external interface models.

  Every attachment has a dedicated BEAM process. Native MLIR work invokes the callbacks from a context worker and
  waits through Kinda's callback runtime; the BEAM scheduler running the callback never executes native MLIR work
  that waits for itself. The external model is owned by the MLIR context. Destroying the context releases the native
  callback state and stops the attachment process.

  Normally attachments are declared with `Beaver.Slang.defop/2`. The interface modules also expose `attach/4` for
  dynamic operation definitions that are not written with Slang.
  """
  use GenServer
  require Logger
  alias Beaver.MLIR
  alias Kinda.CallbackRuntime

  defmodule Attachment do
    @moduledoc "A live callback-backed external interface attachment."
    @enforce_keys [:pid, :id, :interface, :operation_name]
    defstruct [:pid, :id, :interface, :operation_name]
    @type t() :: %__MODULE__{pid: pid(), id: term(), interface: atom(), operation_name: String.t()}
  end

  @type interface() :: :memory_effects | :conditionally_speculatable | :transform_op | :pattern_descriptor
  @registry __MODULE__.Registry
  @doc false
  def global_registrar_child_specs, do: [{Registry, keys: :duplicate, name: @registry}]
  @doc false
  def release_context(%Context{ref: context_ref}, registry \\ @registry) do
    if Process.whereis(registry) do
      for {pid, :attachment} <- Registry.lookup(registry, context_ref) do
        try do
          GenServer.call(pid, :context_destroyed)
        catch
          :exit, {reason, _} when reason in [:noproc, :normal] -> :ok
        end
      end
    end

    :ok
  end

  @doc false
  def start(ctx, attachment, register, dispatch, registry \\ @registry) do
    case GenServer.start(__MODULE__, {ctx, registry, attachment, register, dispatch}) do
      {:ok, pid} -> GenServer.call(pid, :attachment)
      {:error, {%_{} = exception, stacktrace}} -> reraise exception, stacktrace
      {:error, reason} -> raise "failed to attach #{inspect(attachment.__struct__)}: #{inspect(reason)}"
    end
  end

  @doc false
  @spec attach(Context.t(), String.t(), interface(), map(), keyword()) :: Attachment.t()
  def attach(%Context{} = ctx, op_name, interface, callbacks, opts \\ [])
      when is_binary(op_name) and is_map(callbacks) do
    timeout_ms = Beaver.Execution.callback_timeout!(opts)

    start(
      ctx,
      %Attachment{pid: nil, id: nil, interface: interface, operation_name: op_name},
      fn -> native_attach(ctx, op_name, interface, callbacks, timeout_ms) end,
      &dispatch/1
    )
  end

  @doc false
  @spec attach_all(Context.t(), String.t(), [{String.t(), keyword()}], keyword()) :: [Attachment.t()]
  def attach_all(%Context{} = ctx, dialect, declarations, opts \\ []) do
    for {operation, interfaces} <- declarations, {interface, implementation} <- interfaces do
      operation_name = "#{dialect}.#{operation}"

      modules = %{
        memory_effects: MLIR.MemoryEffects,
        conditionally_speculatable: MLIR.ConditionallySpeculatable,
        transform_op: MLIR.TransformOpInterface,
        pattern_descriptor: MLIR.PatternDescriptorOpInterface
      }

      provider =
        modules[interface] ||
          raise ArgumentError, "unsupported external interface #{inspect(interface)} on #{operation_name}"

      provider.attach(ctx, operation_name, implementation, opts)
    end
  end

  @impl true
  def init({ctx, registry, attachment, register, dispatch}) do
    {:ok, _} = Registry.register(registry, ctx.ref, :attachment)
    {id, native_owner} = register.()
    attachment = %{attachment | pid: self(), id: id}
    {:ok, %{context: ctx, attachment: attachment, native_owner: native_owner, dispatch: dispatch}}
  end

  @impl true
  def handle_call(:attachment, _from, state), do: {:reply, state.attachment, state}

  def handle_call(:context_destroyed, _from, state) do
    :ok = CAPI.beaver_raw_external_interface_release(state.native_owner)
    {:stop, :normal, :ok, state}
  end

  @impl true
  def handle_info(:external_interface_released, state), do: {:stop, :normal, state}

  def handle_info(message, state) do
    case state.dispatch.(message) do
      {:handled, _failure} ->
        {:noreply, state}

      :unhandled ->
        Logger.warning("unexpected #{inspect(state.attachment.__struct__)} callback message: #{inspect(message)}")
        {:noreply, state}
    end
  end

  for {interface, name, required, optional} <- [
        {:memory_effects, :memory_effects_attach_fallback_model, :get_effects, []},
        {:conditionally_speculatable, :conditionally_speculatable_attach_fallback_model, :get_speculatability, []},
        {:transform_op, :transform_op_interface_attach_fallback_model, :apply, [:allows_repeated_handle_operands]},
        {:pattern_descriptor, :pattern_descriptor_op_interface_attach_fallback_model, :populate_patterns,
         [:populate_patterns_with_state]}
      ] do
    defp native_attach(ctx, op_name, unquote(interface), callbacks, timeout_ms),
      do:
        apply(
          CAPI,
          unquote(:"beaver_raw_#{name}"),
          [ctx, op_name, Map.fetch!(callbacks, unquote(required))] ++
            Enum.map(unquote(optional), &Map.get(callbacks, &1)) ++ [timeout_ms]
        )
  end

  defp dispatch({:get_effects, token, callback, _id, op, effects}) do
    op = native(op)
    effects = native(effects)

    invoke(
      token,
      fn ->
        specs = invoke_memory_effects(callback, op, effects)
        MLIR.MemoryEffects.append(effects, op, specs)
        {:ok, :ok}
      end,
      &CAPI.beaver_raw_callback_reply/2,
      op
    )
  end

  for {name, arguments, normalize, invoke, reply} <- [
        {:get_speculatability, [:op], :normalize_speculatability, :invoke_reply, :reply_speculatability},
        {:apply, [:op, :rewriter, :results, :state], :normalize_transform_result, :invoke_reply,
         :reply_transform_result},
        {:allows_repeated_handle_operands, [:op], :normalize_boolean, :invoke_reply, :reply_boolean},
        {:populate_patterns, [:op, :patterns], :normalize_void_result, :invoke, :beaver_raw_callback_reply},
        {:populate_patterns_with_state, [:op, :patterns, :state], :normalize_void_result, :invoke,
         :beaver_raw_callback_reply}
      ] do
    args = Enum.map(arguments, &Macro.var(&1, nil))
    op = hd(args)
    extra = if name == :apply, do: [op, Enum.at(args, 2)], else: []
    callback = Macro.var(:callback, nil)
    normalized = {normalize, [], [quote(do: unquote(callback).(unquote_splicing(args))) | extra]}

    normalized =
      if name in [:get_speculatability, :allows_repeated_handle_operands],
        do: quote(do: {:ok, unquote(normalized)}),
        else: normalized

    reply =
      if invoke == :invoke, do: quote(do: &CAPI.beaver_raw_callback_reply/2), else: quote(do: &unquote(reply)(&1, &2))

    defp dispatch({unquote(name), token, unquote(callback), _id, unquote_splicing(args)}) do
      unquote(args) = Enum.map(unquote(args), &native/1)
      unquote(invoke)(token, fn -> unquote(normalized) end, unquote(reply), unquote(op))
    end
  end

  defp dispatch(_message), do: :unhandled

  defp invoke_memory_effects(callback, op, _effects) when is_function(callback, 1),
    do: normalize_effects_result(callback.(op))

  defp invoke_memory_effects(callback, op, effects) when is_function(callback, 2),
    do: normalize_effects_result(callback.(op, effects))

  defp normalize_effects_result(:pure), do: []
  defp normalize_effects_result(:ok), do: []
  defp normalize_effects_result({:ok, effects}) when is_list(effects), do: effects
  defp normalize_effects_result(effects) when is_list(effects), do: effects

  defp normalize_effects_result(other),
    do: raise(ArgumentError, "memory effects callback must return :pure, :ok, or a list, got: #{inspect(other)}")

  defp normalize_speculatability({:ok, value}), do: normalize_speculatability(value)

  defp normalize_speculatability(value)
       when value in [:not_speculatable, :speculatable, :recursively_speculatable], do: value

  defp normalize_speculatability(other),
    do: raise(ArgumentError, "invalid speculatability callback result: #{inspect(other)}")

  defp normalize_transform_result(:ok, _operation, _results), do: {:ok, :success}
  defp normalize_transform_result({:ok, nil}, _operation, _results), do: {:ok, :success}

  defp normalize_transform_result({:ok, mappings}, op, results) do
    MLIR.TransformOpInterface.set_results(results, op, mappings)
    {:ok, :success}
  end

  defp normalize_transform_result(:silenceable_failure, _operation, _results), do: {:ok, :silenceable_failure}
  defp normalize_transform_result({:error, :silenceable}, _operation, _results), do: {:ok, :silenceable_failure}
  defp normalize_transform_result(:definite_failure, _operation, _results), do: {:ok, :definite_failure}
  defp normalize_transform_result({:error, :definite}, _operation, _results), do: {:ok, :definite_failure}

  defp normalize_transform_result(other, _operation, _results),
    do: raise(ArgumentError, "invalid transform apply callback result: #{inspect(other)}")

  defp normalize_boolean({:ok, value}), do: normalize_boolean(value)
  defp normalize_boolean(value) when is_boolean(value), do: value
  defp normalize_boolean(other), do: raise(ArgumentError, "callback must return a boolean, got: #{inspect(other)}")
  defp normalize_void_result({:error, _reason} = error), do: error
  defp normalize_void_result(_value), do: {:ok, :ok}

  for name <- [:invoke, :invoke_reply] do
    defp unquote(name)(token, fun, reply, op) do
      outcome = CallbackRuntime.unquote(name)(token, fun, reply, &diagnose(&1, op, "external interface"))
      {:handled, callback_failure(outcome)}
    end
  end

  @doc false
  def diagnose(outcome, op, label) do
    message =
      case outcome do
        {:exception, kind, reason, stacktrace} ->
          "#{label} callback raised:\n" <> Exception.format(kind, reason, stacktrace)

        {:error, reason} ->
          "#{label} callback failed: #{inspect(reason)}"

        _ ->
          nil
      end

    if message do
      MLIR.Operation.location(op) |> MLIR.Diagnostic.emit(message)
      Logger.error(message)
    end
  end

  for {name, codes, fallback} <- [
        {:reply_speculatability, %{not_speculatable: 0, speculatable: 1, recursively_speculatable: 2}, 0},
        {:reply_transform_result, %{success: 0, silenceable_failure: 1, definite_failure: 2}, 2},
        {:reply_boolean, %{false => 0, true => 1}, 0}
      ] do
    defp unquote(name)(token, {:ok, result}),
      do: CAPI.beaver_raw_callback_reply_code(token, true, Map.fetch!(unquote(Macro.escape(codes)), result))

    defp unquote(name)(token, _failure), do: CAPI.beaver_raw_callback_reply_code(token, false, unquote(fallback))
  end

  defp native(value), do: Beaver.Native.check!(value)
  @doc false
  def callback_failure({:error, reason}), do: {:error, reason}
  def callback_failure({:exception, _kind, _reason, _stacktrace} = exception), do: exception
  def callback_failure(_outcome), do: nil
end
