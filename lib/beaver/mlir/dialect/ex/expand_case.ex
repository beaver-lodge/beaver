defmodule Beaver.MLIR.Dialect.Ex.ExpandCase do
  @moduledoc """
  Expands `ex.case` into nested `ex.if`/`ex.cmp` before dialect conversion.

  A case with clauses `[p1, p2, catch-all]` becomes:

      ex.if ex.cmp(scrutinee, lit(p1), "eq") {
        ... clause 1 body ...
      } else {
        ex.if ex.cmp(scrutinee, lit(p2), "eq") {
          ... clause 2 body ...
        } else {
          ... catch-all body ...
        }
      }

  This is an information-preserving IR-to-IR rewrite inside the `ex`
  universe, so it belongs in the transform layer rather than lowering. Run it
  after `MaterializeBoundVariables` and before
  `Beaver.MLIR.Conversion.Ex.plan/1`.

  The scalar slice supports integer literal patterns (multiple per clause,
  OR-ed together) and an optional guard operand per clause (AND-ed with the
  pattern match, enabling type-bitmap narrowing such as `ex.is_integer`
  predicates). Any other pattern form raises `ArgumentError` explicitly
  instead of being silently dropped.
  """

  alias Beaver.Changeset
  alias Beaver.MLIR
  alias Beaver.MLIR.{IRRewriter, RewriterBase}
  alias Beaver.Walker

  @spec run!(MLIR.Module.t() | MLIR.Operation.t()) :: MLIR.Module.t() | MLIR.Operation.t()
  def run!(%MLIR.Module{} = module) do
    expand_cases(module)
    module
  end

  def run!(%MLIR.Operation{} = operation) do
    expand_cases(operation)
    operation
  end

  # Collect every case in one post-order walk. The previous implementation
  # first walked the complete input to find ex.func operations and then walked
  # each function again, multiplying the BEAM/native traversal boundary cost
  # for large modules. Resolving owners before rewriting preserves the same
  # nested-case-first order while keeping cases outside ex.func untouched.
  defp expand_cases(operation) do
    operation
    |> operations()
    |> Enum.filter(&(MLIR.Operation.name(&1) == "ex.case"))
    |> Enum.flat_map(fn ex_case ->
      case owner_func(ex_case) do
        nil -> []
        owner -> [{ex_case, owner}]
      end
    end)
    |> Enum.each(fn {ex_case, owner} -> expand_case(ex_case, owner) end)
  end

  defp owner_func(operation) do
    operation
    |> Stream.iterate(&MLIR.Operation.parent/1)
    |> Enum.find(fn owner ->
      not MLIR.null?(owner) and MLIR.Operation.name(owner) == "ex.func"
    end)
  end

  defp expand_case(ex_case, owner) do
    context = MLIR.context(ex_case)
    location = MLIR.Operation.location(ex_case)
    [scrutinee] = ex_case |> Walker.operands() |> Enum.to_list()

    result_types =
      ex_case
      |> Walker.results()
      |> Enum.to_list()
      |> Enum.map(&MLIR.Value.type/1)

    [region] = ex_case |> Walker.regions() |> Enum.to_list()
    clauses = region |> Walker.blocks() |> Enum.to_list()

    IRRewriter.with_rewriter(owner, fn rewriter ->
      {top_if, _} =
        RewriterBase.with_insertion_point(rewriter, {:before, ex_case}, fn ->
          top_if = build_chain(clauses, scrutinee, result_types, context, location, rewriter)
          RewriterBase.insert(rewriter, top_if)
          {top_if, :ok}
        end)

      RewriterBase.replace_op(rewriter, ex_case, MLIR.Operation.results(top_if) |> Enum.to_list())
    end)

    :ok
  end

  defp build_chain([clause], scrutinee, result_types, context, location, rewriter) do
    {patterns, guard} = clause_patterns(clause, rewriter)

    if patterns == [] and guard == nil do
      # Catch-all as the only clause: it is a plain body, but still needs a
      # wrapper to carry results, so keep the shape uniform with nested cases.
      build_single_catch_all(clause, result_types, context, location, rewriter)
    else
      build_if(
        clause,
        nil,
        scrutinee,
        {patterns, guard},
        result_types,
        context,
        location,
        rewriter
      )
    end
  end

  defp build_chain([clause | rest], scrutinee, result_types, context, location, rewriter) do
    {patterns, guard} = clause_patterns(clause, rewriter)

    if patterns == [] and guard == nil and rest != [] do
      raise ArgumentError,
            "ex.case catch-all clause must be last, got a clause without patterns before more clauses"
    end

    build_if(
      clause,
      rest,
      scrutinee,
      {patterns, guard},
      result_types,
      context,
      location,
      rewriter
    )
  end

  defp build_if(
         clause,
         rest,
         scrutinee,
         {patterns, guard},
         result_types,
         context,
         location,
         rewriter
       ) do
    cond =
      case {patterns, guard} do
        {[], nil} ->
          raise ArgumentError, "ex.case clause requires at least one integer pattern or a guard"

        {[], _guard} ->
          build_guard_cond(guard, context, location, rewriter)

        {_patterns, nil} ->
          build_cond(scrutinee, patterns, context, location, rewriter)

        {_patterns, _guard} ->
          pattern_cond = build_cond(scrutinee, patterns, context, location, rewriter)
          guard_cond = build_guard_cond(guard, context, location, rewriter)
          build_andi(pattern_cond, guard_cond, context, location, rewriter)
      end

    then_region = move_clause_body(clause, context, location, rewriter)

    else_region =
      build_else(rest, scrutinee, result_types, context, location, rewriter)

    create_if(cond, then_region, else_region, result_types, context, location)
  end

  defp build_single_catch_all(clause, result_types, context, location, rewriter) do
    then_region = move_clause_body(clause, context, location, rewriter)

    # Always-true condition: 0 == 0
    zero = build_lit(0, context, location, rewriter)
    cond = build_cmp(zero, zero, "eq", context, location, rewriter)

    else_region = MLIR.CAPI.mlirRegionCreate()
    block = MLIR.Block.create([], [])
    MLIR.CAPI.mlirRegionAppendOwnedBlock(else_region, block)

    # The else branch is unreachable, but scf.if requires both branches to
    # yield the same number of results, so yield a zero per result (the
    # scalar slice yields i64 results only).
    RewriterBase.with_insertion_point(rewriter, {:start, block}, fn ->
      else_values = List.duplicate(zero, length(result_types))

      yield_op = create_yield(else_values, context, location)
      RewriterBase.insert(rewriter, yield_op)
    end)

    create_if(cond, then_region, else_region, result_types, context, location)
  end

  defp build_else(nil, _scrutinee, _result_types, _context, _location, _rewriter) do
    raise ArgumentError, "ex.case without a catch-all clause is unsupported"
  end

  defp build_else([], _scrutinee, _result_types, _context, _location, _rewriter) do
    raise ArgumentError, "ex.case without a catch-all clause is unsupported"
  end

  defp build_else([clause], scrutinee, result_types, context, location, rewriter) do
    {patterns, guard} = clause_patterns(clause, rewriter)

    if patterns == [] and guard == nil do
      move_clause_body(clause, context, location, rewriter)
    else
      inner =
        build_if(
          clause,
          nil,
          scrutinee,
          {patterns, guard},
          result_types,
          context,
          location,
          rewriter
        )

      wrap_region(inner, result_types, context, location, rewriter)
    end
  end

  defp build_else([clause | rest], scrutinee, result_types, context, location, rewriter) do
    {patterns, guard} = clause_patterns(clause, rewriter)

    if patterns == [] and guard == nil do
      raise ArgumentError,
            "ex.case catch-all clause must be last, got a clause without patterns before more clauses"
    end

    inner =
      build_if(
        clause,
        rest,
        scrutinee,
        {patterns, guard},
        result_types,
        context,
        location,
        rewriter
      )

    wrap_region(inner, result_types, context, location, rewriter)
  end

  defp wrap_region(inner, _result_types, context, location, rewriter) do
    region = MLIR.CAPI.mlirRegionCreate()
    block = MLIR.Block.create([], [])
    MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)

    RewriterBase.with_insertion_point(rewriter, {:start, block}, fn ->
      RewriterBase.insert(rewriter, inner)
      yield_op = create_yield(inner |> Walker.results() |> Enum.to_list(), context, location)
      RewriterBase.insert(rewriter, yield_op)
    end)

    region
  end

  defp clause_patterns(clause, _rewriter) do
    [clause_op | _rest] = clause |> Walker.operations() |> Enum.to_list()

    unless MLIR.Operation.name(clause_op) == "ex.clause" do
      raise ArgumentError,
            "ex.case clause block must start with ex.clause, got #{MLIR.Operation.name(clause_op)}"
    end

    patterns =
      clause_op
      |> Walker.attributes()
      |> then(& &1["patterns"])
      |> case do
        nil -> []
        attribute -> attribute |> Enum.to_list()
      end

    guard =
      case clause_op |> Walker.operands() |> Enum.to_list() do
        [] -> nil
        [guard] -> guard
      end

    {patterns, guard}
  end

  # The match condition for a clause is the OR of its integer literal
  # patterns; an optional guard narrows it further (type-bitmap narrowing).
  defp build_cond(scrutinee, patterns, context, location, rewriter) do
    patterns
    |> Enum.map(&build_eq(scrutinee, &1, context, location, rewriter))
    |> Enum.reduce(fn cond, acc ->
      build_ori(acc, cond, context, location, rewriter)
    end)
  end

  defp build_eq(scrutinee, pattern, context, location, rewriter) do
    lit = build_lit(pattern, context, location, rewriter)
    build_cmp(scrutinee, lit, "eq", context, location, rewriter)
  end

  defp build_guard_cond(guard, context, location, rewriter) do
    zero = build_lit(0, context, location, rewriter)
    build_cmp(guard, zero, "ne", context, location, rewriter)
  end

  defp build_andi(left, right, context, location, rewriter) do
    build_bool_op("arith.andi", left, right, context, location, rewriter)
  end

  defp build_ori(left, right, context, location, rewriter) do
    build_bool_op("arith.ori", left, right, context, location, rewriter)
  end

  defp build_bool_op(op_name, left, right, context, location, rewriter) do
    op =
      %Changeset{name: op_name, context: context, location: location}
      |> Changeset.add_argument([left, right])
      |> Changeset.add_result(MLIR.Type.i64())
      |> MLIR.Operation.create()

    RewriterBase.insert(rewriter, op)
    op |> Walker.results() |> Enum.to_list() |> hd()
  end

  defp build_lit(value, context, location, rewriter) do
    lit_op =
      %Changeset{name: "ex.lit", context: context, location: location}
      |> Changeset.add_argument(value: MLIR.Attribute.integer(MLIR.Type.i64(), value))
      |> Changeset.add_result(MLIR.Type.i64())
      |> MLIR.Operation.create()

    RewriterBase.insert(rewriter, lit_op)
    lit_op |> Walker.results() |> Enum.to_list() |> hd()
  end

  defp build_cmp(left, right, predicate, context, location, rewriter) do
    cmp_op =
      %Changeset{name: "ex.cmp", context: context, location: location}
      |> Changeset.add_argument([left, right])
      |> Changeset.add_argument(predicate: MLIR.Attribute.string(predicate))
      |> Changeset.add_result(MLIR.Type.i64())
      |> MLIR.Operation.create()

    RewriterBase.insert(rewriter, cmp_op)
    cmp_op |> Walker.results() |> Enum.to_list() |> hd()
  end

  defp move_clause_body(clause, _context, _location, rewriter) do
    [clause_op | _body] = clause |> Walker.operations() |> Enum.to_list()

    if MLIR.Operation.name(clause_op) != "ex.clause" do
      raise ArgumentError, "ex.case clause block must start with ex.clause"
    end

    terminator = MLIR.CAPI.mlirBlockGetTerminator(clause)

    if MLIR.null?(terminator) or MLIR.Operation.name(terminator) != "ex.yield" do
      raise ArgumentError, "ex.case clause body must end with ex.yield"
    end

    RewriterBase.erase_op(rewriter, clause_op)

    region = MLIR.CAPI.mlirRegionCreate()
    MLIR.CAPI.mlirBlockDetach(clause)
    MLIR.CAPI.mlirRegionAppendOwnedBlock(region, clause)
    region
  end

  defp create_if(cond, then_region, else_region, result_types, context, location) do
    if_op =
      %Changeset{name: "ex.if", context: context, location: location}
      |> Changeset.add_argument(cond)
      |> Changeset.add_argument(then_region)
      |> Changeset.add_argument(else_region)
      |> Changeset.add_result(result_types)
      |> MLIR.Operation.create()

    if_op
  end

  defp create_yield(values, context, location) do
    %Changeset{name: "ex.yield", context: context, location: location}
    |> Changeset.add_argument(values)
    |> MLIR.Operation.create()
  end

  defp operations(operation) do
    {_, operations} =
      Walker.postwalk(operation, [], fn
        %MLIR.Operation{} = op, acc -> {op, [op | acc]}
        element, acc -> {element, acc}
      end)

    Enum.reverse(operations)
  end
end
