alias Beaver.MLIR.{Block, CAPI, Operation, Value}

defmodule Beaver.MLIR.RewriterBase do
  @moduledoc "This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}."
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Beaver.Native
  alias Beaver.MLIR

  @helpers (for {f, "mlirRewriterBase" <> suffix, arity} <-
                  CAPI.__info__(:functions) |> Enum.map(fn {f, a} -> {f, Atom.to_string(f), a} end) do
              suffix = String.replace_prefix(suffix, "Get", "")
              helper_name = Macro.underscore(suffix)
              args = Macro.generate_arguments(arity, __MODULE__)
              defdelegate unquote(:"#{helper_name}")(unquote_splicing(args)), to: CAPI, as: f
              {helper_name, arity}
            end)

  @extra_helpers for {name, arities} <- [
                       replace: 3..3,
                       replace_op: 3..3,
                       move_region_blocks: 3..3,
                       with_insertion_point: 2..3,
                       emit_op: 3..5,
                       emit_results: 3..5,
                       emit: 3..5,
                       scf_if: 5..6,
                       pack_struct: 4..4,
                       unpack_struct: 3..4
                     ],
                     arity <- arities,
                     do: {name, arity}
  @doc false
  def helpers, do: @extra_helpers ++ @helpers

  @doc false
  def polymorphic_helper_names, do: [:emit_op, :emit_results, :emit, :scf_if, :pack_struct, :unpack_struct]

  @doc """
  Moves all blocks from `source` region to the end of `destination` region through MLIR's rewriter infrastructure
  (`RewriterBase::inlineRegionBefore`).
  """
  @spec move_region_blocks(t(), MLIR.Region.t(), MLIR.Region.t()) :: :ok
  def move_region_blocks(%__MODULE__{} = rewriter, %MLIR.Region{} = source, %MLIR.Region{} = destination) do
    CAPI.beaverRewriterBaseInlineRegionBlocks(rewriter, source, destination)
    :ok
  end

  @doc "Syntactic sugar for `replace_all_uses_with/3` and `replace_all_op_uses_with_operation/3`."

  def replace(%__MODULE__{} = rewriter, %Value{} = from, %Value{} = to), do: replace_all_uses_with(rewriter, from, to)

  def replace(%__MODULE__{} = rewriter, %Operation{} = from, %Operation{} = to),
    do: replace_all_op_uses_with_operation(rewriter, from, to)

  @doc """
  Replaces an operation and erases it through MLIR's rewriter infrastructure.

  A replacement can be another operation, one value, or a list of values. The result counts and types must match
  before MLIR is called.
  """
  @spec replace_op(t(), Operation.t(), Operation.t() | Value.t() | [Value.t()]) :: :ok
  def replace_op(%__MODULE__{} = rewriter, %Operation{} = from, %Operation{} = to) do
    validate_replacement_values!(from, Operation.results(to) |> Enum.to_list())
    replace_op_with_operation(rewriter, from, to)
  end

  def replace_op(%__MODULE__{} = rewriter, %Operation{} = from, %Value{} = to), do: replace_op(rewriter, from, [to])

  def replace_op(%__MODULE__{} = rewriter, %Operation{} = from, values)
      when is_list(values) do
    validate_replacement_values!(from, values)
    replace_op_with_values(rewriter, from, length(values), Beaver.Native.array(values, Value))
  end

  @type insertion_position() ::
          :current
          | :clear
          | {:before, Operation.t()}
          | {:after, Operation.t()}
          | {:after_value, Value.t()}
          | {:start, Block.t()}
          | {:end, Block.t()}

  @doc """
  Runs `fun` and restores the current insertion point afterward.

  Restoration happens on normal return, throw, exit, and exception.
  """
  @spec with_insertion_point(t(), (-> result)) :: result when result: var
  def with_insertion_point(%__MODULE__{} = rewriter, fun) when is_function(fun, 0),
    do: with_insertion_point(rewriter, :current, fun)

  @doc """
  Temporarily moves the insertion point while running `fun`.

  Supported positions are `:clear`, `{:before, operation}`, `{:after, operation}`, `{:after_value, value}`,
  `{:start, block}`, and `{:end, block}`. The previous insertion point is always restored.
  """
  @spec with_insertion_point(t(), insertion_position(), (-> result)) :: result when result: var
  def with_insertion_point(%__MODULE__{} = rewriter, position, fun)
      when is_function(fun, 0) do
    saved = save_insertion_point(rewriter)

    try do
      set_temporary_insertion_point(rewriter, position)
      fun.()
    after
      restore_insertion_point(rewriter, saved)
    end
  end

  defp set_temporary_insertion_point(_rewriter, :current), do: :ok
  defp set_temporary_insertion_point(rewriter, :clear), do: clear_insertion_point(rewriter)

  defp set_temporary_insertion_point(rewriter, {:before, %Operation{} = op}),
    do: set_insertion_point_before(rewriter, op)

  defp set_temporary_insertion_point(rewriter, {:after, %Operation{} = op}), do: set_insertion_point_after(rewriter, op)

  defp set_temporary_insertion_point(rewriter, {:after_value, %Value{} = value}),
    do: set_insertion_point_after_value(rewriter, value)

  defp set_temporary_insertion_point(rewriter, {:start, %Block{} = block}),
    do: set_insertion_point_to_start(rewriter, block)

  defp set_temporary_insertion_point(rewriter, {:end, %Block{} = block}),
    do: set_insertion_point_to_end(rewriter, block)

  defp set_temporary_insertion_point(_rewriter, position),
    do: raise(ArgumentError, "unsupported insertion position: #{inspect(position)}")

  @typedoc "Any rewriter handle convertible to `Beaver.MLIR.RewriterBase`."
  @type rewriter_like() :: t() | MLIR.PatternRewriter.t() | MLIR.ConversionPatternRewriter.t()

  @typedoc "Context and location anchor for operation emission."
  @type anchor() :: Operation.t() | MLIR.Location.t() | {MLIR.Context.t(), MLIR.Location.t()}

  @doc """
  Creates an operation via `Beaver.Changeset`, inserts it at `rewriter`'s current insertion point, and returns the
  created `MLIR.Operation`.
  """
  @spec emit_op(rewriter_like(), anchor(), String.Chars.t(), list(), term()) :: Operation.t()
  def emit_op(rewriter, anchor, op_name, args \\ [], result_types \\ []) do
    base = to_base(rewriter)
    {ctx, loc} = resolve_anchor!(anchor)

    op =
      %Beaver.Changeset{name: to_string(op_name), location: loc, context: ctx}
      |> Beaver.Changeset.add_argument(List.wrap(args))
      |> add_changeset_results(result_types)
      |> Operation.create()

    insert(base, op)
  end

  @doc "Emits `op_name` at `rewriter`'s insertion point and returns all result `MLIR.Value`s as a list."
  @spec emit_results(rewriter_like(), anchor(), String.Chars.t(), list(), term()) :: [Value.t()]
  def emit_results(rewriter, anchor, op_name, args \\ [], result_types \\ []) do
    emit_op(rewriter, anchor, op_name, args, result_types) |> Operation.results() |> Enum.to_list()
  end

  @doc """
  Emits `op_name` at `rewriter`'s insertion point and returns the operation (0 results), the single `MLIR.Value` (1
  result), or `[MLIR.Value.t()]` (multiple results).
  """
  @spec emit(rewriter_like(), anchor(), String.Chars.t(), list(), term()) :: Operation.t() | Value.t() | [Value.t()]
  def emit(rewriter, anchor, op_name, args \\ [], result_types \\ []) do
    op = emit_op(rewriter, anchor, op_name, args, result_types)

    case op |> Operation.results() |> Enum.to_list() do
      [] -> op
      [single] -> single
      many -> many
    end
  end

  @doc """
  Builds and inserts an `scf.if` operation with automatic `scf.yield` termination.

  `then_fn` and optional `else_fn` may have arity 1 (receiving `rewriter`) or arity 0. If a branch callback returns
  a value or list of values without already terminating its block, `scf.yield` is emitted automatically.

  Returns the `scf.if` operation when `result_types == []`, a single `MLIR.Value` when `result_types` is a single
  type, or `[MLIR.Value.t()]` when `result_types` is a list of types.
  """
  @spec scf_if(rewriter_like(), anchor(), Value.t(), term(), function(), function() | nil) ::
          Operation.t() | Value.t() | [Value.t()]
  def scf_if(rewriter, anchor, %Value{} = cond_val, result_types, then_fn, else_fn \\ nil)
      when is_function(then_fn) and (is_nil(else_fn) or is_function(else_fn)) do
    base = to_base(rewriter)
    res_list = List.wrap(result_types)

    if res_list != [] and is_nil(else_fn), do: raise(ArgumentError, "scf_if with results requires an else_fn")

    then_region = MLIR.Region.create()
    else_region = MLIR.Region.create()
    if_op = emit_op(base, anchor, "scf.if", [cond_val, then_region, else_region], res_list)
    populate_scf_branch(base, rewriter, anchor, CAPI.mlirOperationGetRegion(if_op, 0), then_fn)
    if else_fn, do: populate_scf_branch(base, rewriter, anchor, CAPI.mlirOperationGetRegion(if_op, 1), else_fn)
    results = if_op |> Operation.results() |> Enum.to_list()

    cond do
      res_list == [] -> if_op
      is_list(result_types) -> results
      true -> hd(results)
    end
  end

  @doc "Packs `field_values` into an LLVM struct of `struct_type` via `llvm.mlir.undef` and `llvm.insertvalue`."
  @spec pack_struct(rewriter_like(), anchor(), term(), [Value.t()]) :: Value.t()
  def pack_struct(rewriter, anchor, struct_type, field_values) when is_list(field_values) do
    base = to_base(rewriter)
    init = emit(base, anchor, "llvm.mlir.undef", [], struct_type)

    field_values
    |> Enum.with_index()
    |> Enum.reduce(init, fn {%Value{} = val, idx}, acc ->
      emit(base, anchor, "llvm.insertvalue", [acc, val, position: {:dense_i64, [idx]}], struct_type)
    end)
  end

  @doc """
  Extracts fields from `struct_val` via `llvm.extractvalue`.

  `fields` may be `:all` (default, infers all element types from `struct_val`), a non-negative integer count, or an
  explicit list of field types.
  """
  @spec unpack_struct(rewriter_like(), anchor(), Value.t(), :all | non_neg_integer() | [term()]) :: [Value.t()]
  def unpack_struct(rewriter, anchor, %Value{} = struct_val, fields \\ :all) do
    base = to_base(rewriter)
    field_types = resolve_struct_field_types!(struct_val, fields)

    for {ft, idx} <- Enum.with_index(field_types) do
      emit(base, anchor, "llvm.extractvalue", [struct_val, position: {:dense_i64, [idx]}], ft)
    end
  end

  defp resolve_struct_field_types!(_struct_val, field_types) when is_list(field_types), do: field_types

  defp resolve_struct_field_types!(%Value{} = struct_val, count_or_all)
       when count_or_all == :all or (is_integer(count_or_all) and count_or_all >= 0) do
    struct_ty = Value.type(struct_val)

    unless CAPI.mlirTypeIsALLVMStructType(struct_ty) |> Beaver.Native.to_term() do
      raise ArgumentError, "expected !llvm.struct value for unpack_struct, got: #{MLIR.to_string(struct_ty)}"
    end

    total = CAPI.mlirLLVMStructTypeGetNumElementTypes(struct_ty) |> Beaver.Native.to_term()
    n = if count_or_all == :all, do: total, else: count_or_all

    if n > total do
      raise ArgumentError, "requested #{n} struct fields, but #{MLIR.to_string(struct_ty)} only has #{total}"
    end

    for idx <- 0..(n - 1)//1, do: CAPI.mlirLLVMStructTypeGetElementType(struct_ty, idx)
  end

  defp populate_scf_branch(base, orig_rewriter, anchor, region, branch_fn) do
    blk = MLIR.Region.append_block(region)

    ret =
      with_insertion_point(base, {:start, blk}, fn ->
        cond do
          is_function(branch_fn, 2) -> branch_fn.(orig_rewriter, anchor)
          is_function(branch_fn, 1) -> branch_fn.(orig_rewriter)
          is_function(branch_fn, 0) -> branch_fn.()
          true -> raise ArgumentError, "scf_if branch callback must have arity 0, 1, or 2"
        end
      end)

    if is_nil(MLIR.Block.terminator(blk)) do
      yield_vals = ret |> List.wrap() |> Enum.filter(&match?(%Value{}, &1))

      with_insertion_point(base, {:end, blk}, fn -> emit_op(base, anchor, "scf.yield", yield_vals, []) end)
    end
  end

  defp to_base(%__MODULE__{} = base), do: base
  defp to_base(%MLIR.PatternRewriter{} = rw), do: MLIR.PatternRewriter.as_base(rw)
  defp to_base(%MLIR.ConversionPatternRewriter{} = rw), do: MLIR.ConversionPatternRewriter.as_base(rw)

  defp resolve_anchor!(%Operation{} = op), do: {MLIR.context(op), Operation.location(op)}
  defp resolve_anchor!(%MLIR.Location{} = loc), do: {MLIR.context(loc), loc}
  defp resolve_anchor!({%MLIR.Context{} = ctx, %MLIR.Location{} = loc}), do: {ctx, loc}
  defp resolve_anchor!(other), do: raise(ArgumentError, "invalid rewriter anchor: #{inspect(other)}")

  defp add_changeset_results(changeset, :infer), do: Beaver.Changeset.add_result(changeset, :infer)

  defp add_changeset_results(changeset, types),
    do: Enum.reduce(List.wrap(types), changeset, &Beaver.Changeset.add_result(&2, &1))

  defp validate_replacement_values!(op, values) do
    unless Enum.all?(values, &match?(%Value{}, &1)),
      do: raise(ArgumentError, "operation replacements must be MLIR values")

    result_types = op |> Operation.results() |> Enum.map(&Value.type/1)
    replacement_types = Enum.map(values, &Value.type/1)

    unless length(result_types) == length(replacement_types) and
             Enum.zip(result_types, replacement_types)
             |> Enum.all?(fn {result_type, replacement_type} -> MLIR.equal?(result_type, replacement_type) end),
           do: raise(ArgumentError, "operation replacement result counts and types must match")

    :ok
  end
end
