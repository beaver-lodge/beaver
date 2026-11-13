defmodule Beaver.SSA.Flow do
  @moduledoc "Forward facts, liveness and interference-based register layouts over an SSA schema."
  defp ag(a, i), do: :array.get(i, a)
  defp as(a, i, v), do: :array.set(i, v, a)
  defp an(n, d), do: :array.new(n, default: d)
  defp ms(x \\ []), do: MapSet.new(x)
  defp mu(a, b), do: MapSet.union(a, b)
  defp md(a, b), do: MapSet.difference(a, b)
  defp zmap(keys, values), do: Map.new(Enum.zip(keys, values))
  defp rows(list, fun), do: Enum.map(list || [], fun)
  defp fix(fun, state), do: then(fun.(state), fn {next, changed?} -> if(changed?, do: fix(fun, next), else: next) end)
  @doc "Propagates caller-supplied instruction facts and edge selections; exception parameter facts are policy."
  def facts(function, schema, infer, select, exception_facts) do
    fields = schema.schema()
    blocks = List.to_tuple(function[fields.function[:blocks]])

    facts =
      Enum.reduce(
        elem(blocks, 0)[fields.block[:parameters]],
        an(function[fields.function[:values]], :bottom),
        &as(&2, &1, :any)
      )

    f_lp(blocks, facts, as(an(tuple_size(blocks), false), 0, true), {schema, infer, select, exception_facts})
  end

  defp f_lp(bt, fs, ac, {schema, infer, select, exception_facts} = policy) do
    fields = schema.schema()
    kp = &elem(bt, &1)[fields.block[:parameters]]
    ki = fields.block[:instructions]
    ko = fields.instruction[:defs]
    ka = fields.edge[:arguments]
    target = fields.edge[:target]
    exit = fields.block[:exit]
    unwind = fields.block[:unwind]
    p2 = fn d, l, {x, q} -> {x, as(l, d, true), q} end
    uf = fn f, c, k, v, w -> then(Beaver.SSA.Scalar.join(ag(f, k), v, w), &{as(f, k, &1), c or &1 != ag(f, k)}) end
    fi = fn i, s -> Enum.zip_reduce(i[ko], infer.(i, elem(s, 0)), s, fn v, l, {x, d} -> uf.(x, d, v, l, false) end) end
    pp = fn d, a, {f, _} = s, w -> Enum.zip_reduce(kp.(d), a, s, fn p, v, {x, q} -> uf.(x, q, p, ag(f, v), w) end) end
    pe = fn %{^target => d, ^ka => a}, {f, l, c}, i -> p2.(d, l, pp.(d, a, {f, c or !ag(l, d)}, d <= i)) end
    uw = &exception_facts(&1, &2, exception_facts)
    fu = fn h, f, a, c -> if(h, do: {uw.(kp.(h), f), as(a, h, true), c or !ag(a, h)}, else: {f, a, c}) end
    se = fn x, f -> selected_edges(select.(x, f), x, schema) end

    cr =
      &Enum.reduce(schema.exit_values(&1[exit], :defs), Enum.reduce(&1[ki], &2, fi), fn r, {x, d} ->
        uf.(x, d, r, :any, false)
      end)

    f2 = fn i, b, a, {g, d} -> Enum.reduce(se.(b[exit], g), fu.(b[unwind], g, a, d), &pe.(&1, &2, i)) end
    st = fn i, {f, a, c} = s -> if(ag(a, i), do: f2.(i, elem(bt, i), a, cr.(elem(bt, i), {f, c})), else: s) end

    then(Enum.reduce(0..(tuple_size(bt) - 1), {fs, ac, false}, st), fn {g, a, q} ->
      (q && f_lp(bt, g, a, policy)) || :array.to_list(g)
    end)
  end

  defp exception_facts([], facts, _), do: facts

  defp exception_facts(parameters, facts, infer),
    do: Enum.zip_reduce(parameters, infer.(parameters), facts, &as(&3, &1, &2))

  defp selected_edges(:unknown, exit, schema), do: schema.edges(exit)
  defp selected_edges({_, edge}, _, _), do: List.wrap(edge)
  @doc "Drops unreachable blocks and renumbers block and value identifiers in original order."
  def prune(function, schema) do
    fields = schema.schema()
    blocks = List.to_tuple(function[fields.function[:blocks]])

    successors = fn block ->
      Enum.map(schema.edges(block[fields.block[:exit]]), & &1[fields.edge[:target]]) ++
        List.wrap(block[fields.block[:unwind]])
    end

    reachable = reach(blocks, [0], MapSet.new(), successors) |> Enum.sort()
    kept = Enum.map(reachable, &elem(blocks, &1))
    indices = Map.new(Enum.zip(reachable, 0..999_999))
    values = Map.new(Enum.zip(Enum.flat_map(kept, &schema.block_defs/1), 0..999_999))
    reindex = fn edge -> Map.update!(edge, fields.edge[:target], &indices[&1]) end

    blocks =
      for block <- kept do
        block = schema.map_block_values(block, &Map.fetch!(values, &1))
        block = Map.update!(block, fields.block[:exit], &schema.map_edges(&1, reindex))
        if key = fields.block[:unwind], do: Map.put(block, key, indices[block[key]]), else: block
      end

    function |> Map.put(fields.function[:blocks], blocks) |> Map.put(fields.function[:values], map_size(values))
  end

  defp reach(_, [], visited, _), do: visited

  defp reach(blocks, [index | rest], visited, successors) do
    if MapSet.member?(visited, index),
      do: reach(blocks, rest, visited, successors),
      else: reach(blocks, successors.(elem(blocks, index)) ++ rest, MapSet.put(visited, index), successors)
  end

  @doc "Retains effectful instructions and the transitive uses of live results and exits."
  def dead_code(blocks, schema, pure?) do
    fields = schema.schema().block
    live = MapSet.new(Enum.flat_map(blocks, &schema.exit_uses(&1[fields[:exit]])))

    needed? = fn instruction, live ->
      not pure?.(instruction) or Enum.any?(schema.instruction_defs(instruction), &MapSet.member?(live, &1))
    end

    instructions = Enum.flat_map(Enum.reverse(blocks), & &1[fields[:instructions]])

    collect = fn instruction, live ->
      if needed?.(instruction, live),
        do: MapSet.union(live, MapSet.new(schema.instruction_uses(instruction))),
        else: live
    end

    live = fix(fn live -> then(Enum.reduce(instructions, live, collect), &{&1, &1 != live}) end, live)

    Enum.map(
      blocks,
      &Map.update!(&1, fields[:instructions], fn instructions ->
        Enum.filter(instructions, fn i -> needed?.(i, live) end)
      end)
    )
  end

  @doc "Threads caller-approved forwarding blocks; entry blocks and cycles stop traversal."
  def thread_forwarders(blocks, schema, forward) do
    table = List.to_tuple(blocks)
    fields = schema.schema()

    step = fn {edge, seen} ->
      index = edge[fields.edge[:target]]
      next = (index > 0 and not MapSet.member?(seen, index)) && forward.(elem(table, index))
      if next, do: {{next, MapSet.put(seen, index)}, true}, else: {{edge, seen}, false}
    end

    for block <- blocks do
      Map.update!(
        block,
        fields.block[:exit],
        &schema.map_edges(&1, fn edge -> elem(fix(step, {edge, MapSet.new()}), 0) end)
      )
    end
  end

  @doc "Adds deterministic layout, roots, block entries and bounded-work poll points to a function."
  def allocate(f, schema, block_work) do
    fields = schema.schema()
    kp = fields.block[:parameters]
    ki = fields.block[:instructions]
    ko = fields.instruction[:defs]
    kn = fields.instruction[:uses]
    ka = fields.edge[:arguments]
    target = fields.edge[:target]
    exit = fields.block[:exit]
    unwind = fields.block[:unwind]

    {bt, bc, vc, vr} =
      {List.to_tuple(f[fields.function[:blocks]]), length(f[fields.function[:blocks]]), f[fields.function[:values]],
       0..(f[fields.function[:values]] - 1)//1}

    wh = fn b, l -> if(u = b[unwind], do: md(ag(l, u), ms(elem(bt, u)[kp])), else: ms()) end
    wel = fn e, l, a -> a |> mu(md(ag(l, e[target]), ms(elem(bt, e[target])[kp]))) |> mu(ms(e[ka])) end
    wwi = fn h, {i, o}, {y, p} -> then(mu(md(y, ms(i[ko])), mu(ms(i[kn]), h)), &{&1, [{o, &1} | p]}) end

    wx = fn x, l, h ->
      Enum.reduce(schema.edges(x), ms(schema.exit_uses(x)), &wel.(&1, l, &2))
      |> md(ms(schema.exit_values(x, :defs)))
      |> mu(h)
    end

    wl2 = fn s, h, x -> Enum.with_index(s) |> Enum.reverse() |> Enum.reduce({x, [{length(s), x}]}, &wwi.(h, &1, &2)) end
    wl = fn i, l -> then({elem(bt, i), wh.(elem(bt, i), l)}, fn {b, h} -> wl2.(b[ki], h, wx.(b[exit], l, h)) end) end
    lv1 = fn i, {a, q} -> then(elem(wl.(i, a), 0), &if(ag(a, i) == &1, do: {a, q}, else: {as(a, i, &1), true})) end
    lv = fix(&Enum.reduce((bc - 1)..0//-1, {&1, false}, lv1), an(bc, ms()))
    {es, ps} = safepoints(bt, bc, wl, lv, schema, fields, block_work)

    po = fn b, o ->
      (((i = Enum.at(b[ki], o)) && i[ko]) || schema.exit_values(b[exit], :defs)) ++ ((o == 0 && b[kp]) || [])
    end

    u1 = fn s, c -> Enum.reduce(s, c, fn x, a -> Map.update!(a, x, &mu(&1, MapSet.delete(s, x))) end) end
    cf = Enum.reduce(ps, Map.new(vr, &{&1, ms()}), &u1.(ms(po.(elem(bt, &1["block"]), &1["offset"]) ++ &1["live"]), &2))
    {fd, vs} = {fn u -> Enum.find(0..vc, &(&1 not in u)) end, Enum.sort_by(vr, &(-MapSet.size(cf[&1])))}
    la = Enum.reduce(vs, an(vc, 0xFFFFFFFF), &as(&2, &1, fd.(MapSet.new(cf[&1], fn x -> ag(&2, x) end))))
    {ly, rts} = {:array.to_list(la), rows(ps, fn p -> Enum.sort(MapSet.new(p["live"], &ag(la, &1))) end)}
    Map.merge(f, zmap(~w(layout frame_size roots entries points), [ly, Enum.max(ly, fn -> -1 end) + 1, rts, es, ps]))
  end

  defp safepoints(bt, bc, wl, lv, schema, fields, block_work) do
    ki = fields.block[:instructions]
    target = fields.edge[:target]
    exit = fields.block[:exit]
    m0 = fn i, {o, l} -> zmap(~w(block offset live poll), [i, o, Enum.sort(l), false]) end
    e2 = fn i, ps, {e, p, c} -> {[c | e], Enum.reverse(rows(ps, &m0.(i, &1))) ++ p, c + length(ps)} end
    {er, pr, pn} = Enum.reduce(0..(bc - 1)//1, {[], [], 0}, &e2.(&1, elem(wl.(&1, lv), 1), &2))
    {es, et, p0} = {Enum.reverse(er), List.to_tuple(Enum.reverse(er)), :array.from_list(Enum.reverse(pr))}
    {sp, mx} = {&as(&1, &2, %{ag(&1, &2) | "poll" => true}), &as(&1, &2, max(ag(&1, &2), &3))}
    bs = for(i <- 0..(bc - 1)//1, e <- schema.edges(elem(bt, i)[exit]), e[target] <= i, do: elem(et, e[target]))
    {m1, pa} = {&if(&1 > &2, do: mx.(&3, &1, &4), else: &3), Enum.reduce((pn > 0 && [0 | bs]) || [], p0, &sp.(&2, &1))}
    me = fn b, k, m, c -> Enum.reduce(schema.edges(b[exit]), m, &m1.(elem(et, &1[target]), k, &2, c)) end
    p2 = fn k, a, m, b, p, c -> {as(a, k, p), (p["offset"] < length(b[ki]) && mx.(m, k + 1, c)) || me.(b, k, m, c)} end
    p1 = fn k, a, m, d, p -> p2.(k, a, m, elem(bt, p["block"]), p, if(p["poll"], do: 1, else: d + 1)) end
    pd = fn k, {a, m} -> p1.(k, a, m, ag(m, k), ag(if(ag(m, k) >= block_work, do: sp.(a, k), else: a), k)) end
    ps = Enum.reduce(0..(pn - 1)//1, {pa, an(pn, 0)}, pd) |> elem(0) |> :array.to_list()
    {es, ps}
  end
end
