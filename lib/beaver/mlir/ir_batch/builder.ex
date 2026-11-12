defmodule Beaver.MLIR.IRBatch.Builder do
  @moduledoc """
  Immutable, context-independent CFG specifications for `Beaver.MLIR.IRBatch`.

  Values are symbolic bindings, not native handles. `data` belongs to the caller. Integer constants are reused within
  their block, or throughout the function when explicitly emitted in the entry block with `scope: :entry`.
  `finish/1` returns blocks for `IRBatch.build_function!/3`; MLIR verifies types and dominance at submission.

  `expression/2` constructs deferred operands; `emit/2` evaluates them in order against the supplied builder.
  Bind shared results explicitly: reusing an expression describes another emission, including any side effects.
  """
  defstruct blocks: [],
            current: nil,
            args: [],
            operations: [],
            sequence: 0,
            names: MapSet.new(),
            constants: %{},
            block_constants: %{},
            data: %{}

  @predicates Map.new(Enum.with_index(~w(eq ne slt sle sgt sge ult ule ugt uge)a))
  def new(opts \\ []), do: %__MODULE__{data: Map.new(Keyword.get(opts, :data, []))}
  def start_block(b, name, args \\ [])

  def start_block(%__MODULE__{current: nil} = b, name, args) when not is_nil(name) do
    if MapSet.member?(b.names, name), do: raise(ArgumentError, "duplicate block #{inspect(name)}")
    %{b | current: name, args: args, operations: [], block_constants: %{}, names: MapSet.put(b.names, name)}
  end

  def start_block(_, _, _), do: raise(ArgumentError, "terminate the current block before starting another")
  def end_block(%__MODULE__{current: nil}, _), do: raise(ArgumentError, "no current block")

  def end_block(b, terminator) do
    {terminator, b} = emit(b, terminator)
    block = %{name: b.current, body: Enum.reverse([terminator | b.operations])}
    block = if b.args == [], do: block, else: Map.put(block, :args, b.args)
    %{b | blocks: [block | b.blocks], current: nil, args: [], operations: []}
  end

  def finish(%__MODULE__{current: nil, blocks: blocks}), do: Enum.reverse(blocks)
  def finish(_), do: raise(ArgumentError, "unterminated block")
  def fresh(b, prefix \\ "v"), do: {"__#{prefix}_#{b.sequence}", %{b | sequence: b.sequence + 1}}

  def operation(b, name, operands, results, attributes, bind) do
    if is_nil(b.current), do: raise(ArgumentError, "start a block before emitting operations")
    {operands, b} = emit(b, operands)
    op = %{name: name, operands: operands, results: results, attributes: attributes, bind: bind}
    %{b | operations: [op | b.operations]}
  end

  def value(b, name, operands, type, attributes \\ []) do
    {operands, b} = emit(b, operands)
    {v, b} = fresh(b)
    {v, operation(b, name, operands, [type], attributes, v)}
  end

  @doc "Emits func.call; returns `{nil | value | values, builder}` for zero, one or multiple results."
  def call(b, symbol, operands, result_types) do
    {operands, b} = emit(b, operands)
    {values, b} = Enum.map_reduce(result_types, b, fn _, acc -> fresh(acc) end)

    result =
      case values do
        [] -> nil
        [v] -> v
        many -> many
      end

    {result, operation(b, "func.call", operands, result_types, [callee: {:symbol, symbol}], result)}
  end

  @doc "Describes a nested value expression. Each occurrence emits once, in left-to-right operand order."
  def expression(kind, args) when kind in [:constant, :call, :value, :compare, :binary],
    do: {__MODULE__, kind, args}

  @doc "Emits nested expressions in a value, list or tuple, returning the resolved value and immutable builder."
  def emit(b, {__MODULE__, kind, args}) when kind in [:constant, :call, :value, :compare, :binary] do
    {args, b} = emit(b, args)
    apply(__MODULE__, kind, [b | args])
  end

  def emit(b, values) when is_list(values), do: Enum.map_reduce(values, b, &emit(&2, &1))

  def emit(b, values) when is_tuple(values) do
    {values, b} = emit(b, Tuple.to_list(values))
    {List.to_tuple(values), b}
  end

  def emit(b, value), do: {value, b}
  def constant_ref(b, type, value), do: Map.fetch!(b.constants, {type, value})

  def constant(b, type, value, opts \\ []) when is_atom(type) do
    key = {type, value}
    scope = Keyword.get(opts, :scope, :block)
    unless scope in [:block, :entry], do: raise(ArgumentError, "invalid constant scope")

    if scope == :entry and (is_nil(b.current) or b.blocks != []),
      do: raise(ArgumentError, "entry constants must be emitted in the first block")

    cache = if scope == :entry, do: b.constants, else: Map.merge(b.constants, b.block_constants)

    case Map.fetch(cache, key) do
      {:ok, v} ->
        {v, b}

      :error ->
        {v, b} = value(b, "arith.constant", [], type, value: key)
        field = if scope == :entry, do: :constants, else: :block_constants
        {v, Map.update!(b, field, &Map.put(&1, key, v))}
    end
  end

  def compare(b, predicate, lhs, rhs),
    do: value(b, "arith.cmpi", [lhs, rhs], :i1, predicate: {:i64, Map.fetch!(@predicates, predicate)})

  def binary(b, op, operands, type), do: value(b, "arith.#{op}", operands, type)
  @doc "Branches to ready or fallback, then starts ready. The fallback destination can carry block arguments."
  def guard(b, condition, ready, fallback),
    do: b |> end_block({:cond_br, condition, ready, fallback}) |> start_block(ready)

  def join(b, name, types) do
    args = Enum.with_index(types, fn type, i -> {"#{name}_#{i}", type} end)
    {Enum.map(args, &elem(&1, 0)), start_block(b, name, args)}
  end

  @doc "Joins existing fast-path values with values emitted by `slow.(builder)`, returning `{values, builder}`."
  def guarded(b, condition, fast_values, types, slow, opts \\ []) do
    {prefix, b} = if opts[:name], do: {opts[:name], b}, else: fresh(b, "guard")
    slow_name = Keyword.get(opts, :slow, "#{prefix}_slow")
    join_name = Keyword.get(opts, :join, "#{prefix}_join")
    check_results!(fast_values, types)
    b = b |> end_block({:cond_br, condition, {join_name, fast_values}, slow_name}) |> start_block(slow_name)
    {slow_values, b} = slow.(b)
    check_results!(slow_values, types)
    b |> end_block({:br, join_name, slow_values}) |> join(join_name, types)
  end

  defp check_results!(values, types) do
    unless length(values) == length(types), do: raise(ArgumentError, "branch result count does not match join types")
  end

  def switch(b, selector, type, default, cases), do: end_block(b, {:switch, selector, type, default, cases})
  @doc "Emits a switch and each `{case_value, block_name, payload}` arm; `emit.(payload, builder)` must terminate it."
  def dispatch(b, selector, type, default, cases, emit) do
    b = switch(b, selector, type, default, Enum.map(cases, fn {key, name, _} -> {key, name} end))

    Enum.reduce(cases, b, fn {_, name, payload}, acc ->
      result = emit.(payload, start_block(acc, name))
      if result.current, do: raise(ArgumentError, "unterminated dispatch arm #{inspect(name)}")
      result
    end)
  end
end
