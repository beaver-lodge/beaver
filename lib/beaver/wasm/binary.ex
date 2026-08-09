defmodule Beaver.Wasm.Binary do
  @moduledoc """
  A packaged WebAssembly binary with its imports/exports manifest.

  This is the WASM analogue of the `gpu.binary` packaging: the compiled
  binary bytes plus the host-boundary facts (`imports`/`exports`) that a
  runtime or an experiment receipt needs, without re-reading the LLVM IR.

  The manifest is parsed directly from the wasm binary's import/export
  sections, so it reflects what the binary actually declares.
  """

  import Bitwise

  alias Beaver.Wasm.Binary

  @magic <<0x00, 0x61, 0x73, 0x6D>>

  @type import :: %{module: String.t(), name: String.t(), kind: atom()}
  @type export :: %{name: String.t(), kind: atom(), index: non_neg_integer()}

  defstruct [:bytes, :target, :imports, :exports]

  @type t :: %__MODULE__{
          bytes: binary(),
          target: String.t() | nil,
          imports: [import()],
          exports: [export()]
        }

  @doc """
  Parses the wasm binary header and the import/export sections.

  Unknown sections are skipped; a malformed binary raises.
  """
  @spec parse(binary()) :: t()
  def parse(@magic <> <<version::little-32, rest::binary>>) when version != 0 do
    {imports, exports} = parse_sections(rest, [], [])

    %Binary{
      bytes: @magic <> <<version::little-32, rest::binary>>,
      imports: imports,
      exports: exports
    }
  end

  def parse(_other) do
    raise ArgumentError, "not a WebAssembly binary (missing \\0asm magic)"
  end

  defp parse_sections(<<>>, imports, exports), do: {Enum.reverse(imports), Enum.reverse(exports)}

  defp parse_sections(<<id, rest::binary>>, imports, exports) do
    {section_size, payload} = take_uleb(rest)
    <<section::binary-size(^section_size), tail::binary>> = payload

    case id do
      2 -> parse_sections(tail, parse_imports(section, []), exports)
      7 -> parse_sections(tail, imports, parse_exports(section, []))
      _ -> parse_sections(tail, imports, exports)
    end
  end

  # --- imports (section 2) ---
  defp parse_imports(<<0, _::binary>>, acc), do: Enum.reverse(acc)

  defp parse_imports(section, acc) do
    {count, rest} = take_uleb(section)
    parse_import_entries(rest, count, acc)
  end

  defp parse_import_entries(<<>>, 0, acc), do: Enum.reverse(acc)

  defp parse_import_entries(binary, count, acc) when count > 0 do
    {module, rest} = take_name(binary)
    {name, rest} = take_name(rest)
    <<kind, rest::binary>> = rest

    entry = %{module: module, name: name, kind: import_kind(kind)}

    # skip the kind-specific index (function/table/memory/global type index)
    {_index, rest} = take_uleb(rest)
    parse_import_entries(rest, count - 1, [entry | acc])
  end

  defp import_kind(0), do: :func
  defp import_kind(1), do: :table
  defp import_kind(2), do: :memory
  defp import_kind(3), do: :global
  defp import_kind(other), do: {:unknown, other}

  # --- exports (section 7) ---
  defp parse_exports(<<0, _::binary>>, acc), do: Enum.reverse(acc)

  defp parse_exports(section, acc) do
    {count, rest} = take_uleb(section)
    parse_export_entries(rest, count, acc)
  end

  defp parse_export_entries(<<>>, 0, acc), do: Enum.reverse(acc)

  defp parse_export_entries(binary, count, acc) when count > 0 do
    {name, rest} = take_name(binary)
    <<kind, rest::binary>> = rest
    {index, rest} = take_uleb(rest)

    entry = %{name: name, kind: import_kind(kind), index: index}
    parse_export_entries(rest, count - 1, [entry | acc])
  end

  # --- primitives ---
  defp take_name(binary) do
    {len, rest} = take_uleb(binary)
    <<name::binary-size(^len), tail::binary>> = rest
    {name, tail}
  end

  defp take_uleb(binary), do: decode_uleb(binary, 0, 0)

  defp decode_uleb(<<byte, rest::binary>>, acc, shift) do
    value = acc ||| (byte &&& 0x7F) <<< shift

    if (byte &&& 0x80) == 0 do
      {value, rest}
    else
      decode_uleb(rest, value, shift + 7)
    end
  end
end
