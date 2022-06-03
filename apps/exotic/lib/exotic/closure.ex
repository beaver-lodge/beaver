defmodule Exotic.Closure.Definition do
  @enforce_keys [:return_type, :arg_types]
  defstruct [:return_type, :arg_types]
end

defmodule Exotic.Closure.Server do
  use GenServer

  defmodule State do
    @moduledoc """
    State should include the closure because when it is used as a callback, its lifetime couldn't be prolong enough by transmitting.
    """
    @enforce_keys [:callback_module, :def]
    defstruct [:callback_module, :def, :closure, :user_state]
  end

  @impl true
  def init(
        callback_module: callback_module,
        def: %Exotic.Closure.Definition{} = def
      )
      when is_atom(callback_module) do
    {:ok, %State{callback_module: callback_module, def: def}}
  end

  @impl true
  def handle_cast({:closure, closure}, state) do
    {:noreply, %State{state | closure: closure}}
  end

  @impl true
  def handle_call(:user_state, _from, state = %State{user_state: user_state}) do
    {:reply, user_state, state}
  end

  @doc """
    Handling message to call a elixir function from native code.
    Original idea from: https://github.com/tessi/wasmex/issues/256#issuecomment-848339952
  """

  @impl true
  def handle_info(
        {callback_id, arg_values, _ret_ptr_value, token},
        %State{
          callback_module: callback_module,
          def: %Exotic.Closure.Definition{arg_types: arg_types, return_type: _return_type},
          user_state: user_state
        } = state
      )
      when is_atom(callback_id) do
    args =
      for {t, ref} <- Enum.zip(arg_types, arg_values) do
        %Exotic.Value{ref: ref, type: t, holdings: MapSet.new()}
      end

    new_user_state =
      with {:return, ret, new_user_state} <-
             apply(callback_module, :handle_invoke, [callback_id, args, user_state]) do
        _ret_ref = ret |> Exotic.Valuable.resource()
        # TODO: setting the returned value
        Exotic.NIF.finish_callback(token, true)
        new_user_state
      else
        :error ->
          Exotic.NIF.finish_callback(token, true)
          raise "fail to handle callback"

        # Do nothing
        {:pass, new_user_state} ->
          Exotic.NIF.finish_callback(token, true)
          new_user_state

        _ ->
          Exotic.NIF.finish_callback(token, true)
          raise "unexpected result"
      end

    {:noreply, %State{state | user_state: new_user_state}}
  end

  # TODO: implement this in Rust
  # When the closure resource is collected by the Erlang GC, process will receive a message
  @impl true
  def handle_info(
        :drop_closure,
        _state
      ) do
    Process.exit(self(), :normal)
  end
end

defmodule Exotic.Closure do
  @moduledoc """
  A closure is a wrapper for an Elixir function. Backed by a process as the callback handler, it could be called from C by a function pointer.
  It is similar to a Exotic.Function, but it doesn't have a symbol.
  When it is invoked from C (as a function pointer), the arguments will be wrapped as Exotic.Value and callback handler process will receive a message.
  If you want to implement a process of your own, please have a look at `Exotic.Closure.Server`.
  Usually it's a good idea to implement your own GenServer if you want to shared some state between callbacks.
  """

  @enforce_keys [:ref, :pid, :def, :holdings]
  defstruct [:ref, :pid, :def, :holdings]

  @doc """
  Create a closure from a function, a process managed by Exotic will be created to forward the call to the callback handler module.
  """
  def create(
        %__MODULE__.Definition{return_type: return_type, arg_types: arg_types} = def,
        callback_module,
        callback_id
      )
      when is_atom(callback_module) do
    return_type = return_type |> Exotic.Type.get() |> Map.get(:ref)

    {:ok, pid} =
      GenServer.start_link(__MODULE__.Server, callback_module: callback_module, def: def)

    arg_types =
      arg_types
      |> Enum.map(&Exotic.Type.get/1)
      |> Enum.map(&Map.get(&1, :ref))

    closure = %__MODULE__{
      ref: Exotic.NIF.get_closure(pid, callback_id, return_type, arg_types),
      def: def,
      pid: pid,
      holdings: MapSet.new()
    }

    GenServer.cast(pid, {:closure, closure})
    closure
  end

  def create({:ptr, [function: [return_type | arg_types]]}, callback_module, callback_id) do
    definition = %__MODULE__.Definition{return_type: return_type, arg_types: arg_types}
    create(definition, callback_module, callback_id)
  end

  def destroy(%__MODULE__{
        pid: pid
      }) do
    :ok = GenServer.stop(pid)
  end

  def state(%__MODULE__{
        pid: pid
      }) do
    GenServer.call(pid, :user_state)
  end

  @doc """
  You can register a closure a the drop callback for a Exotic.Value. This is useful if you don't want to call some cleanup function everytime.
  """
  def register_drop_callback(_value, _closure = %__MODULE__{}) do
    # TODO: implement this
  end
end
