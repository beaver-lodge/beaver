defmodule Beaver.MLIR.Template do
  @moduledoc """
  Hygienic parameterized MLIR text template rendering and identifier validation.

  Replaces `{{name}}` placeholders with validated MLIR tokens (`{:symbol, sym}`, `{:ssa, name}`, `{:block, label}`,
  `{:ident, id}`, `{:string, text}`, integers, booleans, `MLIR` entities, or `{:raw, fragment}`), rejecting unbound
  keys and unsafe identifier characters before MLIR parsing.
  """

  alias Beaver.MLIR

  @placeholder_regex ~r/\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}/
  @bare_ident_regex ~r/\A[a-zA-Z_$][a-zA-Z0-9_$.$-]*\z/
  @ssa_ident_regex ~r/\A[a-zA-Z0-9_$][a-zA-Z0-9_$.$-]*\z/

  @doc "Validates and formats an MLIR flat symbol token (`@name`)."
  @spec check_symbol!(String.Chars.t()) :: String.t()
  def check_symbol!(name) do
    clean = name |> to_string() |> String.trim_leading("@")

    cond do
      clean == "" ->
        raise ArgumentError, "empty MLIR symbol name"

      Regex.match?(@bare_ident_regex, clean) ->
        "@" <> clean

      String.contains?(clean, ["\n", "\r", <<0>>]) ->
        raise ArgumentError, "invalid control character in MLIR symbol name: #{inspect(name)}"

      true ->
        "@" <> escape_string_literal(clean)
    end
  end

  @doc "Validates and formats an MLIR SSA value identifier (`%name`)."
  @spec check_ssa!(String.Chars.t()) :: String.t()
  def check_ssa!(name), do: check_identifier!(name, "%", @ssa_ident_regex, "SSA identifier")

  @doc "Validates and formats an MLIR block label (`^name`)."
  @spec check_block!(String.Chars.t()) :: String.t()
  def check_block!(name), do: check_identifier!(name, "^", @ssa_ident_regex, "block label")

  @doc "Validates a bare MLIR identifier (without prefix)."
  @spec check_ident!(String.Chars.t()) :: String.t()
  def check_ident!(name), do: check_identifier!(name, "", @bare_ident_regex, "bare identifier")

  defp check_identifier!(name, prefix, regex, label) do
    clean = if prefix == "", do: to_string(name), else: name |> to_string() |> String.trim_leading(prefix)
    unless Regex.match?(regex, clean), do: raise(ArgumentError, "invalid MLIR #{label}: #{inspect(name)}")
    prefix <> clean
  end

  @doc "Escapes an Elixir binary as an MLIR double-quoted string literal."
  @spec escape_string_literal(binary()) :: String.t()
  def escape_string_literal(bin) when is_binary(bin) do
    body =
      for <<byte <- bin>>, into: "" do
        case byte do
          ?\\ -> "\\\\"
          ?" -> "\\\""
          ?\n -> "\\n"
          ?\t -> "\\t"
          b when b in 32..126 -> <<b>>
          b -> "\\" <> (Integer.to_string(b, 16) |> String.pad_leading(2, "0") |> String.downcase())
        end
      end

    "\"" <> body <> "\""
  end

  @doc """
  Renders `template` by substituting `{{key}}` placeholders from `bindings` (keyword list or map).

  Raises `ArgumentError` if any placeholder key is missing from `bindings` or if any unparsed `{{...}}` marker
  remains.
  """
  @spec render!(binary(), map() | keyword()) :: String.t()
  def render!(template, bindings \\ []) when is_binary(template) and (is_map(bindings) or is_list(bindings)) do
    string_map = Map.new(bindings, fn {k, v} -> {to_string(k), v} end)

    rendered =
      Regex.replace(@placeholder_regex, template, fn _whole, key ->
        case Map.fetch(string_map, key) do
          {:ok, val} -> format_binding!(key, val)
          :error -> raise ArgumentError, "missing binding #{inspect(key)} in MLIR.Template.render!/2"
        end
      end)

    if String.contains?(rendered, "{{") or String.contains?(rendered, "}}") do
      raise ArgumentError, "malformed or unclosed template placeholder in MLIR.Template"
    end

    rendered
  end

  @doc "Renders `template` with `bindings` and parses + verifies it as an `MLIR.Module`."
  @spec module!(MLIR.Context.t() | binary(), binary() | map() | keyword(), map() | keyword()) :: MLIR.Module.t()
  def module!(template_or_ctx, bindings_or_template \\ [], opts_or_bindings \\ [])

  def module!(%MLIR.Context{} = ctx, template, bindings) when is_binary(template),
    do: template |> render!(bindings) |> MLIR.Module.create!(ctx: ctx)

  def module!(template, bindings, opts) when is_binary(template),
    do: template |> render!(bindings) |> MLIR.Module.create!(opts)

  defp format_binding!(_key, {tag, sym}) when tag in [:symbol, :sym], do: check_symbol!(sym)
  defp format_binding!(_key, {:ssa, ssa}), do: check_ssa!(ssa)
  defp format_binding!(_key, {tag, blk}) when tag in [:block, :blk], do: check_block!(blk)
  defp format_binding!(_key, {:ident, id}), do: check_ident!(id)
  defp format_binding!(_key, {tag, bin}) when tag in [:string, :str] and is_binary(bin), do: escape_string_literal(bin)
  defp format_binding!(_key, {:int, n}) when is_integer(n), do: Integer.to_string(n)
  defp format_binding!(_key, {:bool, b}) when is_boolean(b), do: if(b, do: "true", else: "false")
  defp format_binding!(_key, {:type, %MLIR.Type{} = ty}), do: MLIR.to_string(ty)
  defp format_binding!(_key, {:type, :ptr}), do: "!llvm.ptr"
  defp format_binding!(_key, {:type, atom}) when is_atom(atom), do: Atom.to_string(atom)
  defp format_binding!(_key, {:type, str}) when is_binary(str), do: str
  defp format_binding!(_key, {:raw, text}) when is_binary(text), do: text
  defp format_binding!(_key, n) when is_integer(n), do: Integer.to_string(n)
  defp format_binding!(_key, f) when is_float(f), do: Float.to_string(f)
  defp format_binding!(_key, b) when is_boolean(b), do: if(b, do: "true", else: "false")

  defp format_binding!(_key, %mod{} = entity)
       when mod in [MLIR.Type, MLIR.Attribute, MLIR.Operation, MLIR.Module, MLIR.Value, MLIR.Identifier],
       do: MLIR.to_string(entity)

  defp format_binding!(_key, text) when is_binary(text), do: text

  defp format_binding!(key, other),
    do: raise(ArgumentError, "unsupported binding value for #{inspect(key)}: #{inspect(other)}")
end
