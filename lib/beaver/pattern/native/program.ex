defmodule Beaver.Pattern.Native.Program do
  @moduledoc """
  A callback-free rewrite program. Predicates support operand/result counts,
  attribute equality, operand/result types and equal operand pairs. Rewrites
  set attributes, replace a single result with an operand, or erase an unused
  operation. Patterns that would make no change report no-match.

      Program.new("llvm.call",
        attributes: [callee: old_callee],
        rewrite: {:attributes, [callee: new_callee]})

  Type and attribute handles bind a program to their owning context. Use
  `Rewrite.with_patterns/3` to compile once and reuse the frozen set.
  """
  alias Beaver.MLIR
  @enforce_keys [:root, :checks, :edits]
  defstruct [:root, :checks, :edits, benefit: 1]
  @type t :: %__MODULE__{root: String.t(), checks: list(), edits: list(), benefit: non_neg_integer()}

  def new(root, opts) when is_binary(root) and byte_size(root) > 0 do
    allowed = [:operands, :results, :attributes, :operand_types, :result_types, :same_operands, :rewrite, :benefit]

    unless Keyword.keyword?(opts) and Keyword.keys(opts) -- allowed == [] and
             length(Keyword.keys(opts)) == length(Enum.uniq(Keyword.keys(opts))),
           do: raise(ArgumentError, "invalid declarative pattern options")

    checks =
      Enum.flat_map(opts, fn
        {kind, n} when kind in [:operands, :results] ->
          [{if(kind == :operands, do: 0, else: 1), index!(n), nil, nil}]

        {:attributes, attrs} ->
          Enum.map(attrs, fn {name, %MLIR.Attribute{ref: ref}} -> {2, 0, to_string(name), ref} end)

        {kind, types} when kind in [:operand_types, :result_types] ->
          type_checks(if(kind == :operand_types, do: 3, else: 4), types)

        {:same_operands, pairs} ->
          Enum.map(pairs, fn {a, b} -> {5, index!(a), nil, index!(b)} end)

        _ ->
          []
      end)

    edits =
      case Keyword.fetch!(opts, :rewrite) do
        {:attributes, [_ | _] = attrs} ->
          names = Enum.map(attrs, fn {name, _} -> to_string(name) end)
          unless length(names) == length(Enum.uniq(names)), do: raise(ArgumentError, "duplicate rewrite attribute")
          Enum.map(attrs, fn {name, %MLIR.Attribute{ref: ref}} -> {0, 0, to_string(name), ref} end)

        {:operand, index} ->
          [{1, index!(index), nil, nil}]

        :erase ->
          [{2, 0, nil, nil}]

        other ->
          raise ArgumentError, "unsupported declarative rewrite: #{inspect(other)}"
      end

    benefit = Keyword.get(opts, :benefit, 1)
    unless is_integer(benefit) and benefit >= 0 and benefit < 65535, do: raise(ArgumentError, "invalid pattern benefit")
    %__MODULE__{root: root, checks: checks, edits: edits, benefit: benefit}
  end

  defp type_checks(kind, types),
    do: Enum.map(types, fn {index, %MLIR.Type{ref: type}} -> {kind, index!(index), nil, type} end)

  defp index!(n) when is_integer(n) and n >= 0, do: n
  defp index!(_), do: raise(ArgumentError, "pattern counts and indices must be non-negative integers")
end
