alias Beaver.MLIR.{CAPI, Context}

defmodule Beaver.MLIR.ExternalInterface do
  @moduledoc """
  Owns callback-backed MLIR external interface models.

  Every attachment has a dedicated BEAM process. Native MLIR work invokes the
  callbacks from a context worker and waits through Kinda's callback runtime;
  the BEAM scheduler running the callback never executes native MLIR work that
  waits for itself. The external model is owned by the MLIR context. Destroying
  the context releases the native callback state and stops the attachment
  process.

  Normally attachments are declared with `Beaver.Slang.defop/2`. The interface
  modules also expose `attach/4` for dynamic operation definitions that are not
  written with Slang.
  """

  use GenServer

  require Logger

  alias Beaver.MLIR
  alias Kinda.CallbackRuntime

  defmodule Attachment do
    @moduledoc "A live callback-backed external interface attachment."
    @enforce_keys [:pid, :id, :interface, :operation_name]
    defstruct [:pid, :id, :interface, :operation_name]

    @type t() :: %__MODULE__{
            pid: pid(),
            id: term(),
            interface: atom(),
            operation_name: String.t()
          }
  end

  @type interface() :: :memory_effects | :conditionally_speculatable | :transform_op | :pattern_descriptor

  @registry __MODULE__.Registry

  @doc false
  def global_registrar_child_specs, do: [{Registry, keys: :duplicate, name: @registry}]

  @doc false
  def release_context(%Context{ref: context_ref}) do
    if Process.whereis(@registry) do
      Registry.dispatch(@registry, context_ref, fn entries ->
        Enum.each(entries, fn {pid, nil} ->
          try do
            GenServer.call(pid, :context_destroyed)
          catch
            :exit, {:noproc, _} -> :ok
            :exit, {:normal, _} -> :ok
          end
        end)
      end)
    end

    :ok
  end

  @doc false
  @spec attach(Context.t(), String.t(), interface(), map(), keyword()) :: Attachment.t()
  def attach(%Context{} = ctx, op_name, interface, callbacks, opts \\ [])
      when is_binary(op_name) and is_map(callbacks) do
    timeout_ms = Keyword.get(opts, :timeout, 30_000)

    unless is_integer(timeout_ms) and timeout_ms >= 0,
      do: raise(ArgumentError, ":timeout must be a non-negative integer")

    init = {ctx, op_name, interface, callbacks, timeout_ms}

    case GenServer.start(__MODULE__, init) do
      {:ok, pid} -> GenServer.call(pid, :attachment)
      {:error, {%_{} = exception, stacktrace}} -> reraise exception, stacktrace
      {:error, reason} -> raise "failed to attach #{interface}: #{inspect(reason)}"
    end
  end

  @doc false
  @spec attach_all(Context.t(), String.t(), [{String.t(), keyword()}], keyword()) ::
          [Attachment.t()]
  def attach_all(%Context{} = ctx, dialect, declarations, opts \\ []) do
    for {operation, interfaces} <- declarations,
        {interface, implementation} <- interfaces do
      operation_name = "#{dialect}.#{operation}"

      case interface do
        :memory_effects ->
          MLIR.MemoryEffects.attach(ctx, operation_name, implementation, opts)

        :conditionally_speculatable ->
          MLIR.ConditionallySpeculatable.attach(ctx, operation_name, implementation, opts)

        :transform_op ->
          MLIR.TransformOpInterface.attach(ctx, operation_name, implementation, opts)

        :pattern_descriptor ->
          MLIR.PatternDescriptorOpInterface.attach(ctx, operation_name, implementation, opts)

        other ->
          raise ArgumentError,
                "unsupported external interface #{inspect(other)} on #{operation_name}"
      end
    end
  end

  @impl true
  def init({ctx, op_name, interface, callbacks, timeout_ms}) do
    {:ok, _} = Registry.register(@registry, ctx.ref, nil)
    {id, native_owner} = native_attach(ctx, op_name, interface, callbacks, timeout_ms)

    {:ok,
     %{
       context: ctx,
       operation_name: op_name,
       interface: interface,
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
      interface: state.interface,
      operation_name: state.operation_name
    }

    {:reply, attachment, state}
  end

  def handle_call(:context_destroyed, _from, state) do
    :ok = CAPI.beaver_raw_external_interface_release(state.native_owner)
    {:stop, :normal, :ok, state}
  end

  @impl true
  def handle_info(:external_interface_released, state), do: {:stop, :normal, state}

  def handle_info(message, state) do
    case dispatch(message) do
      {:handled, _failure} ->
        {:noreply, state}

      :unhandled ->
        Logger.warning("unexpected external interface callback message: #{inspect(message)}")
        {:noreply, state}
    end
  end

  defp native_attach(ctx, op_name, :memory_effects, callbacks, timeout_ms) do
    CAPI.beaver_raw_memory_effects_attach_fallback_model(ctx, op_name, Map.fetch!(callbacks, :get_effects), timeout_ms)
  end

  defp native_attach(ctx, op_name, :conditionally_speculatable, callbacks, timeout_ms) do
    CAPI.beaver_raw_conditionally_speculatable_attach_fallback_model(
      ctx,
      op_name,
      Map.fetch!(callbacks, :get_speculatability),
      timeout_ms
    )
  end

  defp native_attach(ctx, op_name, :transform_op, callbacks, timeout_ms) do
    CAPI.beaver_raw_transform_op_interface_attach_fallback_model(
      ctx,
      op_name,
      Map.fetch!(callbacks, :apply),
      Map.get(callbacks, :allows_repeated_handle_operands),
      timeout_ms
    )
  end

  defp native_attach(ctx, op_name, :pattern_descriptor, callbacks, timeout_ms) do
    CAPI.beaver_raw_pattern_descriptor_op_interface_attach_fallback_model(
      ctx,
      op_name,
      Map.fetch!(callbacks, :populate_patterns),
      Map.get(callbacks, :populate_patterns_with_state),
      timeout_ms
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

  defp dispatch({:get_speculatability, token, callback, _id, op}) do
    op = native(op)

    invoke_reply(token, fn -> {:ok, normalize_speculatability(callback.(op))} end, &reply_speculatability/2, op)
  end

  defp dispatch({:apply, token, callback, _id, op, rewriter, results, state}) do
    op = native(op)
    rewriter = native(rewriter)
    results = native(results)
    state = native(state)

    invoke_reply(
      token,
      fn -> normalize_transform_result(callback.(op, rewriter, results, state), op, results) end,
      &reply_transform_result/2,
      op
    )
  end

  defp dispatch({:allows_repeated_handle_operands, token, callback, _id, op}) do
    op = native(op)

    invoke_reply(token, fn -> {:ok, normalize_boolean(callback.(op))} end, &reply_boolean/2, op)
  end

  defp dispatch({:populate_patterns, token, callback, _id, op, patterns}) do
    op = native(op)
    patterns = native(patterns)

    invoke(token, fn -> callback.(op, patterns) |> normalize_void_result() end, &CAPI.beaver_raw_callback_reply/2, op)
  end

  defp dispatch({:populate_patterns_with_state, token, callback, _id, op, patterns, state}) do
    op = native(op)
    patterns = native(patterns)
    state = native(state)

    invoke(
      token,
      fn -> callback.(op, patterns, state) |> normalize_void_result() end,
      &CAPI.beaver_raw_callback_reply/2,
      op
    )
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

  defp invoke(token, fun, reply, op) do
    outcome = CallbackRuntime.invoke(token, fun, reply, &diagnose(&1, op))
    {:handled, callback_failure(outcome)}
  end

  defp invoke_reply(token, fun, reply, op) do
    outcome = CallbackRuntime.invoke_reply(token, fun, reply, &diagnose(&1, op))
    {:handled, callback_failure(outcome)}
  end

  defp diagnose({:exception, kind, reason, stacktrace}, op) do
    message =
      "external interface callback raised:\n" <> Exception.format(kind, reason, stacktrace)

    MLIR.Operation.location(op) |> MLIR.Diagnostic.emit(message)
    Logger.error(message)
  end

  defp diagnose({:error, reason}, op) do
    message = "external interface callback failed: #{inspect(reason)}"
    MLIR.Operation.location(op) |> MLIR.Diagnostic.emit(message)
    Logger.error(message)
  end

  defp diagnose(_outcome, _operation), do: :ok

  defp reply_speculatability(token, {:ok, value}) do
    code =
      %{not_speculatable: 0, speculatable: 1, recursively_speculatable: 2}
      |> Map.fetch!(value)

    CAPI.beaver_raw_callback_reply_code(token, true, code)
  end

  defp reply_speculatability(token, _failure), do: CAPI.beaver_raw_callback_reply_code(token, false, 0)

  defp reply_transform_result(token, {:ok, result}) do
    code = %{success: 0, silenceable_failure: 1, definite_failure: 2} |> Map.fetch!(result)
    CAPI.beaver_raw_callback_reply_code(token, true, code)
  end

  defp reply_transform_result(token, _failure), do: CAPI.beaver_raw_callback_reply_code(token, false, 2)

  defp reply_boolean(token, {:ok, value}),
    do: CAPI.beaver_raw_callback_reply_code(token, true, if(value, do: 1, else: 0))

  defp reply_boolean(token, _failure), do: CAPI.beaver_raw_callback_reply_code(token, false, 0)

  defp native(value), do: Beaver.Native.check!(value)

  defp callback_failure({:error, reason}), do: {:error, reason}

  defp callback_failure({:exception, _kind, _reason, _stacktrace} = exception), do: exception

  defp callback_failure(_outcome), do: nil
end
