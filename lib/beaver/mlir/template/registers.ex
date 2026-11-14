defmodule Beaver.MLIR.Template.Registers do
  @moduledoc """
  Generates standard-dialect templates for a three-word tagged register ABI: boxing,
  checked scalar arithmetic, state access, bounded binary views and helper dispatch.
  The caller supplies types, tags, symbol prefixes, arithmetic names, service indices and C bindings.
  State fields follow `(ptr, ptr, ptr, i64, i64, i64, i64, i64, ptr, i64, ptr, i64)`; the binary opcode family is fixed.
  Integer, boolean, nil and float tags form a contiguous predicate range; cursor offsets occupy the high tag bits.
  """
  alias Beaver.MLIR.{CABI, IRBatch}
  alias IRBatch.{Builder, Expressions}
  alias Builder, as: B
  import Expressions
  @runtime_inputs [state: :ptr, pc: :i64, index: :i64, args: :ptr, outputs: :ptr]
  defmacro __using__(opts) do
    quote do
      @registers Map.new(unquote(opts))
      @runtime_symbols Enum.map(~w(helper cold), &("#{@registers.prefix}#{@registers.runtime_prefix}_" <> &1))
      @reserved [@registers.prefix <> "poll" | @runtime_symbols]
      @inline_prefixes [@registers.prefix, "#{@registers.runtime_prefix}.value."]
      @signature_prefixes ["#{@registers.runtime_prefix}.raw." | @inline_prefixes]
      def arithmetic_specs, do: @registers.arithmetic
      def comparison_specs, do: @registers.comparisons
      def term_type, do: @registers.term
      def resume_state_type, do: @registers.state
      def inline_source, do: unquote(__MODULE__).inline_source(@registers)
      def runtime_source, do: unquote(__MODULE__).runtime_source(@registers)

      use Beaver.MLIR.Template.Library,
        libraries: [inline: :private, runtime: :preserve],
        dependencies: [unquote(__MODULE__), @registers.runtime, CABI, CABI.Provider, Builder, Expressions, IRBatch],
        signatures: &signature_symbol?/1,
        templates: &template_symbol?/2

      defp template_symbol?(name, :inline), do: inline_template_symbol?(name)
      defp template_symbol?(name, :runtime), do: name in @runtime_symbols
      defp inline_template_symbol?(name), do: name not in @reserved and String.starts_with?(name, @inline_prefixes)
      defp signature_symbol?(name), do: String.starts_with?(name, @signature_prefixes)
    end
  end

  defp view_inputs(abi), do: [input: abi.term, cache: :ptr, ctx: :ptr, table: :ptr, args: :ptr, outputs: :ptr]
  defp fun(abi, name, inputs, results, body), do: function(abi.prefix <> name, inputs, results, body)
  defp reg(abi, name, args, types \\ nil), do: call(abi.prefix <> name, args, types || abi.term)
  defp box(abi, kind, value), do: reg(abi, "box_#{kind}", [value])
  defp field(value, index), do: val("llvm.extractvalue", [value], :i64, position: {:dense_i64, [index]})
  defp tag(abi, value), do: arith("andi", field(value, 0), int(abi.tag_mask))

  defp pack(abi, words) do
    Enum.reduce(Enum.with_index(words), val("llvm.mlir.undef", [], abi.term), fn {v, i}, s ->
      val("llvm.insertvalue", [s, v], abi.term, position: {:dense_i64, [i]})
    end)
  end

  defp core_specs(abi) do
    intrinsic = call("#{abi.runtime_prefix}_intrinsic_call", [:state, :index, :args, :outputs], :i32)

    simple = [
      fun(abi, "box", [tag: :i64, bits: :i64, aux: :i64], abi.term, pack(abi, [:tag, :bits, :aux])),
      fun(abi, "term_tag", [t: abi.term], :i64, tag(abi, :t)),
      fun(abi, "suspend", [s: :i32], :i32, :s),
      fun(abi, "intrinsic", [state: :ptr, index: :i64, args: :ptr, outputs: :ptr], :i32, intrinsic),
      fun(abi, "binary_cursor", [bin: abi.term, delta: :i64], abi.term, fn b ->
        {w0, b} = B.emit(b, field(:bin, 0))

        offset =
          select(
            cmp(:eq, arith("andi", w0, int(abi.tag_mask)), int(abi.tags.cursor)),
            arith("shrui", w0, int(abi.cursor_shift)),
            int(0)
          )

        advanced = arith("shli", arith("addi", offset, :delta), int(abi.cursor_shift))
        cursor = arith("ori", advanced, int(abi.tags.cursor))
        ret(b, pack(abi, [cursor, field(:bin, 1), field(:bin, 2)]))
      end),
      fun(abi, "try_predicate", [v: abi.term, expected_tag: :i64], [abi.term, :i1], [
        box(abi, :bool_i1, cmp(:eq, field(:v, 0), :expected_tag)),
        cmp(:ule, arith("subi", field(:v, 0), int(abi.tags.integer)), int(abi.tags.float - abi.tags.integer))
      ]),
      fun(abi, "truth", [v: abi.term], abi.term, fn b ->
        {t, b} = B.emit(b, tag(abi, :v))

        false? = all([cmp(:eq, t, int(abi.tags.boolean)), cmp(:eq, field(:v, 1), int(0))])
        falsy = arith("ori", false?, cmp(:eq, t, int(abi.tags.nil)), :i1)
        ret(b, box(abi, :bool_i1, arith("xori", falsy, int(1, :i1), :i1)))
      end),
      fun(abi, "load_byte_at", [base_i64: :i64, off: :i64], [:i64, :ptr], fn b ->
        {ptr, b} = B.emit(b, gep(val("llvm.inttoptr", [:base_i64], :ptr), :off, :i8))
        ret(b, [cast("extui", val("llvm.load", [ptr], :i8), :i64), ptr])
      end),
      byte_spec(abi)
    ]

    boxes =
      for {name, inputs, tag, bits} <- [
            {"int", [bits: :i64], abi.tags.integer, :bits},
            {"bool", [bits: :i64], abi.tags.boolean, :bits},
            {"bool_i1", [cond: :i1], abi.tags.boolean, cast("extui", :cond, :i64)},
            {"nil", [], abi.tags.nil, int(0)},
            {"empty", [], abi.tags.empty, int(0)}
          ],
          do: fun(abi, "box_#{name}", inputs, abi.term, pack(abi, [int(tag), bits, int(0)]))

    fields =
      for {name, i} <- [{"word0", 0}, {"bits", 1}, {"aux", 2}],
          do: fun(abi, "term_#{name}", [t: abi.term], :i64, field(:t, i))

    shapes =
      for {name, expected, condition} <- [
            {"nil", abi.tags.nil, nil},
            {"boolean", abi.tags.boolean, nil},
            {"true", abi.tags.boolean, :ne},
            {"false", abi.tags.boolean, :eq}
          ] do
        check = cmp(:eq, tag(abi, :v), int(expected))
        check = if condition, do: all([check, cmp(condition, field(:v, 1), int(0))]), else: check
        fun(abi, "shape_#{name}", [v: abi.term], abi.term, box(abi, :bool_i1, check))
      end

    negations =
      for {prefix, types} <- [{"scalar", [abi.term]}, {"try_int", [abi.term, :i1]}] do
        body = reg(abi, "#{prefix}_sub", [box(abi, :int, int(0)), :a], types)
        fun(abi, "#{prefix}_neg", [a: abi.term], types, body)
      end

    simple ++ negations ++ boxes ++ fields ++ shapes
  end

  defp byte_spec(abi) do
    fun(abi, "byte", [bin: abi.term], [abi.term, abi.term], fn b ->
      {w0, b} = B.emit(b, field(:bin, 0))
      {t, b} = B.emit(b, arith("andi", w0, int(abi.tag_mask)))

      {offset, b} =
        B.emit(b, select(cmp(:eq, t, int(abi.tags.cursor)), arith("shrui", w0, int(abi.cursor_shift)), int(0)))

      {view, b} = B.emit(b, call("#{abi.runtime_prefix}_binary_view", [:bin], abi.term))

      valid =
        all([
          arith("ori", cmp(:eq, t, int(abi.tags.heap)), cmp(:eq, t, int(abi.tags.cursor)), :i1),
          cmp(:ne, field(view, 0), int(0)),
          cmp(:ult, offset, field(view, 2))
        ])

      address = arith("addi", field(view, 1), offset)
      byte = cast("extui", val("llvm.load", [val("llvm.inttoptr", [address], :ptr)], :i8), :i64)

      branches(b, valid, [box(abi, :int, byte), box(abi, :bool, int(1))], [
        box(abi, :int, int(0)),
        box(abi, :bool, int(0))
      ])
    end)
  end

  defp math_specs(abi) do
    arithmetic =
      for {name, op} <- abi.arithmetic, do: {name, op, if(op in ~w(addi subi muli), do: :wide, else: :integer)}

    comparisons = for {name, pred} <- abi.comparisons, do: {"cmp_#{name}", String.to_existing_atom(pred), :cmp}

    for operation = {_, _, family} <- arithmetic ++ [{"div_rem", "divsi", :pair}] ++ comparisons,
        variant <-
          Enum.map(~w(int scalar try_int), &{&1, false}) ++
            if(family == :wide, do: [{"int", true}, {"try_int", true}], else: []),
        do: math_spec(abi, operation, variant)
  end

  defp math_spec(abi, {name, op, family}, {prefix, unboxed?}) do
    try? = prefix == "try_int"
    checked? = prefix != "scalar"
    types = List.duplicate(abi.term, if(family == :pair, do: 2, else: 1))
    symbol = "#{prefix}_#{name}#{if unboxed?, do: "_i64"}"
    inputs = if unboxed?, do: [x: :i64, y: :i64], else: [a: abi.term, b: abi.term]

    recipe = %{
      name: name,
      operation: if(family == :pair, do: "div_rem", else: op),
      family: family,
      try?: try?,
      checked?: checked?,
      unboxed?: unboxed?,
      types: types,
      fallback?: checked? and (not try? or family not in [:wide, :cmp])
    }

    fun(abi, symbol, inputs, types ++ if(try?, do: [:i1], else: []), &math_body(abi, recipe, &1))
  end

  defp math_body(abi, recipe, b) do
    {[x, y], b} = B.emit(b, if(recipe.unboxed?, do: [:x, :y], else: [field(:a, 1), field(:b, 1)]))
    tag = fn v -> if recipe.try?, do: field(v, 0), else: tag(abi, v) end

    ints =
      if recipe.unboxed?,
        do: int(1, :i1),
        else: all([cmp(:eq, tag.(:a), int(abi.tags.integer)), cmp(:eq, tag.(:b), int(abi.tags.integer))])

    {values, safe, b} = signed_binary(b, recipe.operation, x, y, recipe.checked?)
    {ok, b} = if recipe.checked?, do: B.emit(b, all([ints, safe])), else: {nil, b}
    fast = Enum.map(values, &box(abi, if(recipe.family == :cmp, do: :bool_i1, else: :int), &1))
    valid = if recipe.try?, do: [ok], else: []
    if recipe.fallback?, do: math_fallback(abi, recipe, b, {x, y}, {fast, ok, valid}), else: ret(b, fast ++ valid)
  end

  defp math_fallback(abi, recipe, b, {x, y}, {fast, ok, valid}) do
    slow_name =
      if recipe.family == :cmp,
        do: "#{abi.runtime_prefix}_cmp_slow_" <> String.replace_prefix(recipe.name, "cmp_", ""),
        else: "#{abi.runtime_prefix}_int_slow_#{recipe.name}"

    args = if recipe.unboxed?, do: [box(abi, :int, x), box(abi, :int, y)], else: [:a, :b]

    slow =
      if recipe.try?,
        do: List.duplicate(box(abi, :int, int(0)), length(recipe.types)),
        else: call(slow_name, args, recipe.types)

    b = B.end_block(b, {:cond_br, ok, :yes, :no}) |> B.start_block(:yes) |> ret(fast ++ valid) |> B.start_block(:no)
    {slow, b} = B.emit(b, slow)
    ret(b, List.wrap(slow) ++ valid)
  end

  defp indirect(callee, args),
    do:
      val("llvm.call", [callee | args], :i32,
        op_bundle_sizes: {:dense_i32, []},
        operandSegmentSizes: {:dense_i32, [length(args) + 1, 0]}
      )

  defp host(index),
    do: indirect(load(:table, int(index), :ptr), [:ctx, val("llvm.mlir.zero", [], :ptr), :args, :outputs])

  defp next(b, destination, values, name, args \\ []),
    do: b |> B.end_block({:br, destination, values}) |> B.start_block(name, args)

  defp state_ptr(abi, index),
    do:
      val("llvm.getelementptr", [:state], :ptr,
        elem_type: {:type, abi.state},
        rawConstantIndices: {:dense_i32, [0, index]}
      )

  defp state_load(abi, index, type), do: val("llvm.load", [state_ptr(abi, index)], type)

  defp state_specs(abi) do
    reads =
      for {type, indices} <- [ptr: [0, 1, 2, 8, 10], i64: [3, 4, 5, 6, 7, 9, 11]],
          i <- indices,
          do: fun(abi, "state_#{type}_#{i}", [state: :ptr], type, state_load(abi, i, type))

    writes =
      for i <- [4, 5, 6, 9] do
        body = &ret(store(&1, state_ptr(abi, i), :val), [])
        fun(abi, "set_state_i64_#{i}", [state: :ptr, val: :i64], [], body)
      end

    reads ++ writes
  end

  def runtime_source(abi) do
    specs =
      for name <- ~w(helper cold) do
        callee = load(state_load(abi, 2, :ptr), :index, :ptr)
        args = [state_load(abi, 1, :ptr), load(state_load(abi, 8, :ptr), :pc, :ptr), :args, :outputs]

        attrs = [
          "llvm.linkage": {:parse, "#llvm.linkage<internal>"},
          "llvm.passthrough": {:parse, if(name == "cold", do: ~s(["noinline", "cold"]), else: ~s(["noinline"]))}
        ]

        spec = fun(abi, "#{abi.runtime_prefix}_#{name}", @runtime_inputs, :i32, indirect(callee, args))
        Map.put(spec, :attributes, attrs)
      end

    "module {\n" <> source(specs) <> "\n}"
  end

  defp budget_spec(abi) do
    fun(abi, "budget_gate", [fuel: :ptr, ckpt: :ptr, ctx: :ptr, table: :ptr, args: :ptr, outputs: :ptr], :i1, fn b ->
      {fuel, b} = B.emit(b, load(:fuel, int(0), :i64))
      b = B.guard(b, cmp(:ugt, fuel, int(0)), :gate, :stop)

      b =
        B.end_block(b, {:cond_br, cmp(:ugt, fuel, load(:ckpt, int(0), :i64)), :go, :check}) |> B.start_block(:check)

      {status, b} = B.emit(b, host(abi.services.checkpoint))

      checkpoint = select(cmp(:ugt, fuel, int(abi.quantum)), arith("subi", fuel, int(abi.quantum)), int(0))
      b = store(b, gep(:ckpt, int(0), :i64), checkpoint)

      b =
        B.end_block(b, {:cond_br, cmp(:eq, status, int(0, :i32)), :go, :stop})
        |> B.start_block(:go)
        |> ret(int(1, :i1))

      ret(B.start_block(b, :stop), int(0, :i1))
    end)
  end

  defp view_spec(abi, checked?) do
    fun(abi, "binary_view_#{if checked?, do: "checked", else: "pos"}", view_inputs(abi), abi.term, fn b ->
      {[w0, w1, w2], b} = B.emit(b, for(i <- 0..2, do: field(:input, i)))
      {[d0, d1, d2, d3, d4], b} = B.emit(b, for(i <- 0..4, do: load(:cache, int(i), :i64)))
      {t, b} = B.emit(b, arith("andi", w0, int(abi.tag_mask)))
      {cursor, b} = B.emit(b, cmp(:eq, t, int(abi.tags.cursor)))
      {binary, b} = B.emit(b, arith("ori", cmp(:eq, t, int(abi.tags.heap)), cursor, :i1))

      hit =
        all([
          cmp(if(checked?, do: :ne, else: :eq), d0, int(if(checked?, do: 0, else: 1))),
          cmp(:eq, d1, w1),
          cmp(:eq, d2, w2),
          binary
        ])

      b =
        B.end_block(b, {:cond_br, hit, {:join, [if(checked?, do: d0, else: int(1)), d3, d4]}, :miss})
        |> B.start_block(:miss)

      b = store(b, gep(:args, int(0), abi.term), :input)
      {_, b} = B.emit(b, host(abi.services.binary_view))
      {view, b} = B.emit(b, load(:outputs, int(0), abi.term))
      {[m0, m1, m2], b} = B.emit(b, for(i <- 0..2, do: field(view, i)))
      status = if checked?, do: select(binary, select(cmp(:ne, m0, int(0)), int(1), int(2)), int(0)), else: m0

      b =
        Enum.with_index([status, w1, w2, m1, m2])
        |> Enum.reduce(b, fn {v, i}, b -> store(b, gep(:cache, int(i), :i64), v) end)

      b = next(b, :join, [m0, m1, m2], :join, j0: :i64, j1: :i64, j2: :i64)
      {offset, b} = B.emit(b, select(cursor, arith("shrui", w0, int(abi.cursor_shift)), int(0)))

      valid = cast("extui", all([cmp(:ule, offset, :j2), cmp(:eq, :j0, int(1))]), :i64)
      ret(b, pack(abi, [valid, arith("addi", :j1, offset), arith("subi", :j2, offset)]))
    end)
  end

  defp binary_spec(abi) do
    inputs =
      [op: :i32, input: abi.term, index: abi.term, expected: :i64] ++ Keyword.drop(view_inputs(abi), [:input])

    fun(abi, "try_binary", inputs, [abi.term, abi.term, abi.term, :i1], fn b ->
      {empty, b} = B.emit(b, reg(abi, "box_empty", []))
      {[yes, no], b} = B.emit(b, [box(abi, :bool, int(1)), box(abi, :bool, int(0))])
      child = arith("ori", cmp(:eq, :op, int(4, :i32)), cmp(:eq, :op, int(5, :i32)), :i1)
      b = B.end_block(b, {:cond_br, child, :child, :position}) |> B.start_block(:child)
      args = Keyword.keys(view_inputs(abi))
      b = next(b, :join, [reg(abi, "binary_view_checked", args)], :position)
      b = next(b, :join, [reg(abi, "binary_view_pos", args)], :join, view: abi.term)
      {[tag, address, size], b} = B.emit(b, for(i <- 0..2, do: field(:view, i)))
      {has_index, b} = B.emit(b, cmp(:uge, :op, int(7, :i32)))
      {bit_read, b} = B.emit(b, cmp(:uge, :op, int(8, :i32)))
      {at, b} = B.emit(b, select(has_index, field(:index, 1), int(0)))
      {offset, b} = B.emit(b, select(bit_read, arith("shrui", at, int(3)), at))
      length_only = arith("ori", cmp(:eq, :op, int(0, :i32)), cmp(:eq, :op, int(3, :i32)), :i1)
      limit = select(cmp(:eq, :op, int(9, :i32)), cmp(:ule, offset, size), cmp(:ult, offset, size), :i1)

      valid =
        all([
          cmp(:ne, tag, int(0)),
          select(has_index, cmp(:eq, field(:index, 0), int(abi.tags.integer)), int(1, :i1), :i1),
          select(bit_read, cmp(:eq, arith("andi", at, int(7)), int(0)), int(1, :i1), :i1),
          select(length_only, int(1, :i1), limit, :i1)
        ])

      b = B.guard(b, valid, :fast, :bad)

      arms = Enum.with_index(~w(shape shape load size load child_result load load load tail load)a, &{&2, &1})

      b = B.switch(b, :op, :i32, :bad, arms)

      results = [
        shape: [yes],
        size: [box(abi, :int, size)],
        child_result: [reg(abi, "binary_cursor", [:input, int(1)])],
        tail: [
          reg(abi, "binary_cursor", [:input, offset]),
          box(abi, :int, arith("shli", size, int(3))),
          yes
        ]
      ]

      b = Enum.reduce(results, b, fn {name, values}, b -> binary_return(B.start_block(b, name), values, empty) end)
      {[byte, ptr], b} = B.emit(B.start_block(b, :load), reg(abi, "load_byte_at", [address, offset], [:i64, :ptr]))
      b = B.switch(b, :op, :i32, :number, [{2, :expected}, {6, :byte}, {8, :integer}, {10, :utf8}])
      number = box(abi, :int, byte)

      results = [
        expected: [box(abi, :bool_i1, cmp(:eq, byte, :expected))],
        number: [number],
        byte: [number, yes],
        integer: [number, box(abi, :int, arith("addi", at, int(8))), yes]
      ]

      b = Enum.reduce(results, b, fn {name, values}, b -> binary_return(B.start_block(b, name), values, empty) end)

      decode = reg(abi, "utf8_decode", [byte, ptr, arith("subi", size, offset)], [:i64, :i64, :i1])
      {[cp, used, valid], b} = B.emit(B.start_block(b, :utf8), decode)

      b =
        B.guard(b, valid, :decoded, :bad)
        |> binary_return([box(abi, :int, cp), box(abi, :int, arith("addi", at, used)), yes], empty)

      b = B.start_block(b, :bad) |> B.end_block({:cond_br, cmp(:ule, :op, int(2, :i32)), :false_shape, :bad_byte})
      b = binary_return(B.start_block(b, :false_shape), [no], empty)

      b =
        B.start_block(b, :bad_byte) |> B.end_block({:cond_br, cmp(:eq, :op, int(6, :i32)), :false_byte, :fallback})

      b = binary_return(B.start_block(b, :false_byte), [box(abi, :int, int(0)), no], empty)
      ret(B.start_block(b, :fallback), [empty, empty, empty, int(0, :i1)])
    end)
  end

  defp binary_return(b, values, empty),
    do: ret(b, values ++ List.duplicate(empty, 3 - length(values)) ++ [int(1, :i1)])

  def inline_source(abi) do
    helpers = for arity <- 0..4, do: {"helper_#{arity}", ["i64", "i64"] ++ List.duplicate(abi.term, arity), 1}
    shapes = for kind <- abi.heap_shapes, do: {"shape_#{kind}", [abi.term, "i64"], 1}

    bits = [
      bits_start: {0, 1},
      bits_finish: {2, 1},
      read_bits_2: {2, 3},
      read_bits_3: {3, 3},
      append_bits_3: {3, 2},
      append_bits_4: {4, 2}
    ]

    bits = for {name, {ins, outs}} <- bits, do: {to_string(name), List.duplicate(abi.term, ins), outs}

    forwards =
      for {name, types, count} <- helpers ++ shapes ++ bits do
        callee = if String.starts_with?(name, "helper_"), do: "guest_" <> name, else: name
        forward(abi.prefix <> name, "#{abi.runtime_prefix}_#{callee}", types, List.duplicate(abi.term, count))
      end

    memory =
      Enum.flat_map([{"slot", abi.term}, {"word", :i64}], fn {name, type} ->
        memory_accessors("#{abi.prefix}load_#{name}", "#{abi.prefix}store_#{name}", type)
      end)

    runtime = {~w(!llvm.ptr i64 i64 !llvm.ptr !llvm.ptr), ["i32"]}

    declarations = [
      {"#{abi.prefix}poll", {["i64"], ["i1"]}},
      {"#{abi.prefix}#{abi.runtime_prefix}_helper", runtime},
      {"#{abi.prefix}#{abi.runtime_prefix}_cold", runtime}
    ]

    wrappers =
      for name <- abi.runtime.functions(),
          do:
            CABI.wrapper_spec(
              abi.runtime.value_symbol(name),
              abi.runtime.raw_symbol(name),
              abi.runtime.value_parameters(name),
              return_type: "i32"
            )

    specs =
      core_specs(abi) ++
        math_specs(abi) ++
        state_specs(abi) ++
        Enum.map([false, true], &view_spec(abi, &1)) ++
        [budget_spec(abi), utf8_decoder("#{abi.prefix}utf8_decode"), binary_spec(abi)] ++ memory ++ forwards

    IO.iodata_to_binary(["module {\n", source(specs ++ wrappers, declarations), "}\n"])
  end
end
