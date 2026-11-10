defmodule Beaver.MLIR.Safepoint do
  @moduledoc """
  Native safepoint liveness analysis, conflict-free frame slot coloring, and
  resumable CFG state machine lowering backed by MLIR's `Liveness` and `DominanceInfo`.
  """

  alias Beaver.{MLIR, Native}
  alias Beaver.MLIR.{CAPI, Operation}

  @type safepoint_entry() :: %{
          id: non_neg_integer(),
          operation: Operation.t(),
          live_values: [MLIR.Value.t()],
          slots: [non_neg_integer()]
        }

  @type analysis_result() :: %{
          slot_count: non_neg_integer(),
          slot_types: [MLIR.Type.t()],
          value_slots: %{MLIR.Value.t() => non_neg_integer()},
          safepoints: [safepoint_entry()]
        }

  @doc "Computes live-across-safepoint SSA values inside `func_op` and assigns conflict-free frame slots."
  @spec analyze(Operation.t(), [Operation.t() | String.t()] | String.t(), keyword()) :: analysis_result()
  def analyze(%Operation{} = func_op, safepoints, opts \\ []) when is_list(opts) do
    exclude_entry_args = Keyword.get(opts, :exclude_entry_args, false)
    ops = resolve_ops(func_op, safepoints)
    CAPI.beaver_raw_safepoint_analyze(func_op.ref, Enum.map(ops, & &1.ref), exclude_entry_args) |> decode()
  end

  @doc "Lowers `func_op` in-place into a resumable coroutine / state-machine CFG."
  @spec lower_resumable(Operation.t(), keyword()) :: {:ok, Operation.t()} | {:error, term()}
  def lower_resumable(%Operation{} = func_op, opts \\ []) when is_list(opts) do
    safepoint_spec = Keyword.get(opts, :safepoints, ["coro.suspend"])
    save_op = Keyword.get(opts, :save_op, "coro.save") |> to_string()
    load_op = Keyword.get(opts, :load_op, "coro.load") |> to_string()

    state_arg =
      case Keyword.get(opts, :state_arg, 0) do
        nil -> -1
        idx when is_integer(idx) -> idx
      end

    ops = resolve_ops(func_op, safepoint_spec)

    try do
      :ok = CAPI.beaver_raw_safepoint_lower_resumable(func_op.ref, Enum.map(ops, & &1.ref), save_op, load_op, state_arg)
      {:ok, func_op}
    rescue
      e -> {:error, e}
    end
  end

  @doc "Lowers `func_op` in-place into a resumable state-machine CFG, raising on failure."
  @spec lower_resumable!(Operation.t(), keyword()) :: Operation.t()
  def lower_resumable!(%Operation{} = func_op, opts \\ []) do
    case lower_resumable(func_op, opts) do
      {:ok, op} -> op
      {:error, reason} -> raise ArgumentError, "failed to lower resumable CFG: #{inspect(reason)}"
    end
  end

  @doc false
  def resolve_ops(%Operation{} = func_op, op_name) when is_binary(op_name), do: resolve_ops(func_op, [op_name])
  def resolve_ops(%Operation{} = _func_op, [%Operation{} | _] = ops), do: ops
  def resolve_ops(%Operation{} = _func_op, []), do: []

  def resolve_ops(%Operation{} = func_op, names) when is_list(names) do
    name_set = MapSet.new(names, &to_string/1)

    {_node, reversed} =
      Beaver.Walker.prewalk(func_op, [], fn
        %Operation{} = op, acc ->
          if not MLIR.equal?(op, func_op) and MapSet.member?(name_set, Operation.name(op)),
            do: {op, [op | acc]},
            else: {op, acc}

        other, acc ->
          {other, acc}
      end)

    Enum.reverse(reversed)
  end

  @doc false
  def decode({slot_count, raw_slot_types, raw_value_slots, raw_entries}) do
    %{
      slot_count: slot_count,
      slot_types: Enum.map(raw_slot_types, &Native.check!/1),
      value_slots: Map.new(raw_value_slots, fn {val, slot} -> {Native.check!(val), slot} end),
      safepoints:
        Enum.map(raw_entries, fn {id, op, live_values, slots} ->
          %{id: id, operation: Native.check!(op), live_values: Enum.map(live_values, &Native.check!/1), slots: slots}
        end)
    }
  end
end
