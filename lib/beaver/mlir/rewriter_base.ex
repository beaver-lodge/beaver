alias Beaver.MLIR.{Block, CAPI, Operation, Value}

defmodule Beaver.MLIR.RewriterBase do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
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

  @doc false
  def helpers,
    do: [
      {:replace, 3},
      {:replace_op, 3},
      {:move_region_blocks, 3},
      {:with_insertion_point, 2},
      {:with_insertion_point, 3}
      | @helpers
    ]

  @doc """
  Moves all blocks from `source` region to the end of `destination` region through
  MLIR's rewriter infrastructure (`RewriterBase::inlineRegionBefore`).
  """
  @spec move_region_blocks(t(), MLIR.Region.t(), MLIR.Region.t()) :: :ok
  def move_region_blocks(%__MODULE__{} = rewriter, %MLIR.Region{} = source, %MLIR.Region{} = destination) do
    CAPI.beaverRewriterBaseInlineRegionBlocks(rewriter, source, destination)
    :ok
  end

  @doc """
  Syntactic sugar for `replace_all_uses_with/3` and `replace_all_op_uses_with_operation/3`.
  """

  def replace(%__MODULE__{} = rewriter, %Value{} = from, %Value{} = to), do: replace_all_uses_with(rewriter, from, to)

  def replace(%__MODULE__{} = rewriter, %Operation{} = from, %Operation{} = to),
    do: replace_all_op_uses_with_operation(rewriter, from, to)

  @doc """
  Replaces an operation and erases it through MLIR's rewriter infrastructure.

  A replacement can be another operation, one value, or a list of values. The
  result counts and types must match before MLIR is called.
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

  Supported positions are `:clear`, `{:before, operation}`,
  `{:after, operation}`, `{:after_value, value}`, `{:start, block}`, and
  `{:end, block}`. The previous insertion point is always restored.
  """
  @spec with_insertion_point(t(), insertion_position(), (-> result)) :: result
        when result: var
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
