defmodule Beaver.MLIR.Triton.LayoutAudit do
  @moduledoc """
  Structural audit of `ttg.convert_layout` operations in Triton GPU IR.

  Triton's layout semantics live in the tensor type encoding
  (`#ttg.blocked<...>`, `#ttg.shared<...>`, `#ttg.dot_operand<...>`, ...).
  A `ttg.convert_layout` marks a point where data movement between encodings
  may happen. This module collects every such operation from a module,
  records its source/target encodings, the tensor types involved, and the
  source location, and renders a stable, sorted report without requiring a
  GPU or a running Triton pipeline.

  The audit intentionally keeps unknown attributes and type text intact
  instead of silently dropping them: the report carries the raw MLIR text for
  anything it cannot classify.

  Requires a context with Triton dialects registered (`Beaver.Triton.register/1`)
  before the module is parsed.
  """

  alias Beaver.MLIR

  @type layout() :: %{
          kind: String.t(),
          params: String.t() | nil,
          raw: String.t()
        }

  @type convert_layout() :: %{
          index: non_neg_integer(),
          location: String.t(),
          source_type: String.t(),
          target_type: String.t(),
          source_layout: layout(),
          target_layout: layout()
        }

  @type t() :: %__MODULE__{
          operation_count: non_neg_integer(),
          convert_layouts: [convert_layout()]
        }

  defstruct [:operation_count, :convert_layouts]

  @convert_layout_op "ttg.convert_layout"

  @doc """
  Collects all `ttg.convert_layout` operations from a module, in stable
  (location, index) order.
  """
  @spec audit(MLIR.Module.t()) :: t()
  def audit(%MLIR.Module{} = module) do
    {_, convert_layouts} =
      module
      |> MLIR.Module.body()
      |> Beaver.Walker.prewalk([], fn
        %MLIR.Operation{} = operation, acc ->
          if MLIR.Operation.name(operation) == @convert_layout_op do
            {operation, [convert_layout_fact(operation, length(acc)) | acc]}
          else
            {operation, acc}
          end

        other, acc ->
          {other, acc}
      end)

    %__MODULE__{
      operation_count: length(convert_layouts),
      convert_layouts: convert_layouts |> Enum.reverse() |> Enum.sort_by(&{&1.location, &1.index})
    }
  end

  @doc "Returns the audit as a deterministic list of maps, newest first by location."
  @spec to_list(t()) :: [convert_layout()]
  def to_list(%__MODULE__{convert_layouts: convert_layouts}), do: convert_layouts

  @doc "Returns the distinct layout kinds seen by the audit, in first-seen order."
  @spec layout_kinds(t()) :: [String.t()]
  def layout_kinds(%__MODULE__{convert_layouts: convert_layouts}) do
    convert_layouts
    |> Enum.flat_map(&[&1.source_layout.kind, &1.target_layout.kind])
    |> Enum.uniq()
  end

  @doc """
  Parses a `#ttg.<kind><{params}>` layout encoding from a tensor type text.

  Falls back to classifying the whole text as unknown instead of dropping it.
  """
  @spec parse_layout(String.t()) :: layout()
  def parse_layout(type_text) when is_binary(type_text) do
    case Regex.run(~r/#ttg\.([a-z_]+)(<\{.*\}>)?/, type_text) do
      [_, kind, params] ->
        %{kind: kind, params: params, raw: "#ttg.#{kind}#{params}"}

      [_, kind] ->
        %{kind: kind, params: nil, raw: "#ttg.#{kind}"}

      _ ->
        %{kind: "unknown", params: nil, raw: type_text}
    end
  end

  @doc """
  Extracts the layout encoding from a `tensor<...xT, #ttg.<...>>` type text.

  The layout is the last comma-separated part of the tensor type; when it is
  missing, the whole text is treated as an unclassified layout.
  """
  @spec extract_layout(String.t()) :: layout()
  def extract_layout(type_text) when is_binary(type_text) do
    case String.split(type_text, ", ", parts: 2) do
      [_shape, encoding] -> parse_layout(encoding)
      _ -> parse_layout(type_text)
    end
  end

  @doc "Extracts `{shape, element_type}` facts from a tensor type text."
  @spec type_facts(String.t()) :: %{shape: String.t() | nil, element_type: String.t() | nil}
  def type_facts(type_text) when is_binary(type_text) do
    case Regex.run(~r/^tensor<(.+?)(?:, .+)?>$/s, type_text) do
      [_, inner] ->
        case Regex.named_captures(~r/^(?<shape>\d+(?:x\d+)*)x(?<element>.+)$/, inner) do
          %{"shape" => shape, "element" => element} ->
            %{shape: shape, element_type: element}

          _ ->
            %{shape: nil, element_type: inner}
        end

      _ ->
        %{shape: nil, element_type: nil}
    end
  end

  defp convert_layout_fact(operation, index) do
    [source_result, target_result] = MLIR.Operation.results(operation) |> Enum.to_list()

    source_type = source_result |> MLIR.Value.type() |> MLIR.to_string()
    target_type = target_result |> MLIR.Value.type() |> MLIR.to_string()

    %{
      index: index,
      location: operation |> MLIR.location() |> MLIR.to_string(),
      source_type: source_type,
      target_type: target_type,
      source_layout: extract_layout(source_type),
      target_layout: extract_layout(target_type),
      source_facts: type_facts(source_type),
      target_facts: type_facts(target_type)
    }
  end
end
