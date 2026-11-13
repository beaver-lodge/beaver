defmodule Beaver.Test.DocumentationExample do
  @moduledoc false
  @root Path.expand("../..", __DIR__)
  def read!(document, marker) when is_binary(document) and is_binary(marker) do
    path = Path.expand(document, @root)
    markdown = File.read!(path)
    tag = "<!--[ \\t]*beaver:test:#{Regex.escape(marker)}[ \\t]*-->"

    case Regex.scan(~r/^#{tag}[ \t]*\r?$/m, markdown) do
      [] -> raise ArgumentError, "missing beaver:test:#{marker} marker in #{path}"
      [_] -> :ok
      _ -> raise ArgumentError, "duplicate beaver:test:#{marker} marker in #{path}"
    end

    pattern = ~r/^#{tag}[ \t]*\r?\n(?:[ \t]*\r?\n)*```elixir[ \t]*\r?\n(.*?)^```[ \t]*\r?$/ms

    case Regex.run(pattern, markdown, capture: :all_but_first) do
      [code] ->
        code

      _ ->
        raise ArgumentError,
              "expected an Elixir code fence after beaver:test:#{marker} in #{path}"
    end
  end
end
