Mix.install([{:zig_parser, "~> 0.1.0"}])

defmodule Updater do
  def run do
    translate_out =
      IO.stream(:stdio, :line)
      |> Stream.map(&String.trim/1)
      |> Enum.join()

    zig_ast =
      Zig.Parser.parse(translate_out).code
      |> dbg
  end
end

Updater.run()
