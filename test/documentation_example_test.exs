defmodule DocumentationExampleTest do
  use ExUnit.Case, async: true

  alias Beaver.Test.DocumentationExample

  @moduletag :smoke

  @examples [
    {"README.md", "quickstart"},
    {"guides/blocks.md", "control-flow"},
    {"guides/passes-and-patterns.md", "pass-definition"},
    {"guides/passes-and-patterns.md", "pass-pipeline"}
  ]

  for {document, marker} <- @examples do
    test "#{document}:#{marker} is present and valid Elixir syntax" do
      code = DocumentationExample.read!(unquote(document), unquote(marker))
      assert {:ok, _} = Code.string_to_quoted(code, file: unquote(document))
    end
  end

  @tag :tmp_dir
  test "reads only the requested block from an explicit document path", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "examples.md")
    File.write!(path, example("first", "1 + 1") <> "\n" <> example("second", "2 + 2"))
    assert DocumentationExample.read!(path, "second") == "2 + 2\n"
  end

  @tag :tmp_dir
  test "escapes marker names and accepts blank lines and CRLF", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "examples.md")
    File.write!(path, "<!-- beaver:test:example.v1 -->\r\n\r\n```elixir\r\n:ok\r\n```\r\n")
    assert DocumentationExample.read!(path, "example.v1") == ":ok\r\n"

    assert_raise ArgumentError, ~r/missing beaver:test:exampleXv1/, fn ->
      DocumentationExample.read!(path, "exampleXv1")
    end
  end

  @tag :tmp_dir
  test "rejects missing and duplicate markers", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "examples.md")
    File.write!(path, example("repeated", ":ok") <> "\n<!-- beaver:test:repeated -->\n")

    assert_raise ArgumentError, ~r/missing beaver:test:absent/, fn -> DocumentationExample.read!(path, "absent") end

    assert_raise ArgumentError, ~r/duplicate beaver:test:repeated/, fn ->
      DocumentationExample.read!(path, "repeated")
    end
  end

  @tag :tmp_dir
  test "does not skip prose or another marker to find a later code fence", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "examples.md")
    File.write!(path, "<!-- beaver:test:missing-code -->\nNot code.\n" <> example("other", ":ok"))

    assert_raise ArgumentError, ~r/expected an Elixir code fence after beaver:test:missing-code/, fn ->
      DocumentationExample.read!(path, "missing-code")
    end
  end

  @tag :tmp_dir
  test "rejects non-Elixir and unclosed code fences", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "examples.md")

    for fence <- ["```text\n:ok\n```\n", "```elixir\n:ok\n"] do
      File.write!(path, "<!-- beaver:test:invalid -->\n" <> fence)

      assert_raise ArgumentError, ~r/expected an Elixir code fence after beaver:test:invalid/, fn ->
        DocumentationExample.read!(path, "invalid")
      end
    end
  end

  defp example(marker, code), do: "<!-- beaver:test:#{marker} -->\n```elixir\n#{code}\n```\n"
end
