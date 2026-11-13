defmodule Beaver.MLIR.IRBatch.Expressions do
  @moduledoc """
  Deferred standard-dialect expressions for `IRBatch.Builder`. Import alongside the builder or a CFG DSL.
  Expressions emit in operand order on every use; explicitly bind calls and loads before sharing them.
  Function bodies can be expressions or builder callbacks. `source/2` infers missing func.call declarations,
  checks signatures and verifies the generated IR before rendering it as module-body text.
  `signed_binary/5` results are deferred: branch on their safety predicate before emitting a division or shift.
  `utf8_decoder/1` expects a loaded first byte and the remaining byte count, including that byte.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.IRBatch.Builder, as: B

  def function(name, inputs, results, body) do
    builder = B.new() |> B.start_block(:entry)
    builder = if is_function(body, 1), do: body.(builder), else: ret(builder, body)
    %{name: name, inputs: inputs, results: List.wrap(results), visibility: :private, blocks: B.finish(builder)}
  end

  @doc "Forwards all arguments and results to a callee, preserving their order."
  def forward(name, callee, types, results) do
    inputs = Enum.with_index(types, &{"a#{&2}", &1})
    function(name, inputs, results, call(callee, Enum.map(inputs, &elem(&1, 0)), results))
  end

  @doc "Builds indexed load/store functions for an element type."
  def memory_accessors(read, write, type) do
    inputs = [base: :ptr, index: :i64]

    [
      function(read, inputs, type, load(:base, :index, type)),
      function(write, inputs ++ [value: type], [], &ret(store(&1, gep(:base, :index, type), :value), []))
    ]
  end

  def source(specs, declarations \\ []) do
    MLIR.Context.with_context(fn ctx ->
      MLIR.Module.with_module(MLIR.IRBatch.build_module!(ctx, specs, verify: false), fn module ->
        calls = Beaver.Walker.find_ops(module, "func.call") |> Enum.chunk_every(256)
        calls = Enum.flat_map(calls, &MLIR.IRBatch.read(&1, attributes: [:callee]))
        signatures = Enum.map(calls, &signature/1)
        MLIR.SymbolTable.ensure_declarations!(module, declarations ++ signatures)
        MLIR.verify!(module)
        module |> MLIR.Module.body() |> MLIR.Block.operations() |> Enum.map_join("\n", &MLIR.to_string/1)
      end)
    end)
  end

  defp signature(call) do
    types = &Enum.map(&1, fn type -> MLIR.to_string(type) end)
    {MLIR.Attribute.value(call.attributes["callee"]), {types.(call.operand_types), types.(call.result_types)}}
  end

  def val(name, args, type, attrs \\ []), do: B.expression(:value, [name, args, type, attrs])
  def int(n, type \\ :i64)
  def int(n, :i128), do: val("arith.constant", [], :i128, value: {:parse, "#{n} : i128"})
  def int(n, type), do: B.expression(:constant, [type, n])
  def arith(name, a, b, type \\ :i64), do: B.expression(:binary, [name, [a, b], type])
  def cmp(pred, a, b), do: B.expression(:compare, [pred, a, b])
  def all([value | rest]), do: Enum.reduce(rest, value, &arith("andi", &2, &1, :i1))
  def select(c, a, b, type \\ :i64), do: val("arith.select", [c, a, b], type)
  def cast(op, value, type), do: val("arith." <> op, [value], type)
  def call(name, args, types), do: B.expression(:call, [name, args, List.wrap(types)])

  def gep(base, index, type),
    do:
      val("llvm.getelementptr", [base, index], :ptr,
        elem_type: {:type, type},
        rawConstantIndices: {:dense_i32, [-2_147_483_648]}
      )

  def load(base, index, type), do: val("llvm.load", [gep(base, index, type)], type)
  def store(b, ptr, value), do: B.operation(b, "llvm.store", [value, ptr], [], [], nil)

  def ret(b, values) do
    {values, b} = B.emit(b, values)
    B.end_block(b, {:return, List.wrap(values)})
  end

  def branches(b, condition, yes, no) do
    b = B.end_block(b, {:cond_br, condition, :yes, :no})
    b = ret(B.start_block(b, :yes), yes)
    ret(B.start_block(b, :no), no)
  end

  @doc "Signed i64 expressions plus a safety predicate. Bind the predicate and branch before consuming division or shift results."
  def signed_binary(b, op, x, y, checked? \\ true) do
    min = -9_223_372_036_854_775_808
    wide? = checked? and op in ~w(addi subi muli)

    {wide, b} =
      if wide?, do: B.emit(b, arith(op, cast("extsi", x, :i128), cast("extsi", y, :i128), :i128)), else: {nil, b}

    safe =
      cond do
        wide? ->
          all([cmp(:sge, wide, int(min, :i128)), cmp(:sle, wide, int(-min - 1, :i128))])

        op in ~w(divsi remsi div_rem) ->
          all([cmp(:ne, y, int(0)), arith("ori", cmp(:ne, x, int(min)), cmp(:ne, y, int(-1)), :i1)])

        op in ~w(shli shrsi shrui) ->
          cmp(:ult, y, int(64))

        true ->
          int(1, :i1)
      end

    {binary_results(op, x, y, wide), safe, b}
  end

  defp binary_results(_, _, _, wide) when not is_nil(wide), do: [cast("trunci", wide, :i64)]
  defp binary_results(op, x, y, _) when is_atom(op), do: [cmp(op, x, y)]
  defp binary_results("div_rem", x, y, _), do: [arith("divsi", x, y), arith("remsi", x, y)]
  defp binary_results(op, x, y, _), do: [arith(op, x, y)]
  @doc "A bounded UTF-8 decoder returning `{codepoint, bits_consumed, valid}`; checks length before each load."
  def utf8_decoder(name) do
    function(name, [byte: :i64, addr: :ptr, rem: :i64], [:i64, :i64, :i1], fn b ->
      b = B.end_block(b, {:cond_br, cmp(:ule, :byte, int(127)), {:ok, [:byte, int(8), int(1, :i1)]}, "m2"})
      widths = [{2, 194, 223, 31, 128}, {3, 224, 239, 15, 2048}, {4, 240, 244, 7, 65_536}]
      b = Enum.reduce(widths, b, &utf8_width/2)

      b =
        B.start_block(b, :bad)
        |> B.end_block({:br, :ok, [int(0), int(0), int(0, :i1)]})
        |> B.start_block(:ok, cp: :i64, bits: :i64, valid: :i1)

      ret(b, [:cp, :bits, :valid])
    end)
  end

  defp utf8_width({width, low, high, mask, minimum}, b) do
    b = B.start_block(b, "m#{width}")
    b = if width < 4, do: B.guard(b, cmp(:ule, :byte, int(high)), "w#{width}", "m#{width + 1}"), else: b
    valid = all([cmp(:ule, arith("subi", :byte, int(low)), int(high - low)), cmp(:uge, :rem, int(width))])
    b = B.guard(b, valid, "load#{width}", :bad)

    {{cp, tails}, b} =
      Enum.reduce(1..(width - 1), {{arith("andi", :byte, int(mask)), []}, b}, fn i, {{cp, tails}, b} ->
        {byte, b} = B.emit(b, cast("extui", load(:addr, int(i), :i8), :i64))
        cp = arith("ori", arith("shli", cp, int(6)), arith("andi", byte, int(63)))
        {{cp, tails ++ [cmp(:eq, arith("andi", byte, int(192)), int(128))]}, b}
      end)

    {cp, b} = B.emit(b, cp)
    allowed = utf8_allowed(width, cp)

    B.end_block(
      b,
      {:cond_br, all([cmp(:uge, cp, int(minimum)), allowed | tails]), {:ok, [cp, int(width * 8), int(1, :i1)]}, :bad}
    )
  end

  defp utf8_allowed(2, _), do: int(1, :i1)
  defp utf8_allowed(3, cp), do: cmp(:ugt, arith("subi", cp, int(55_296)), int(2047))
  defp utf8_allowed(4, cp), do: cmp(:ule, cp, int(1_114_111))
end
