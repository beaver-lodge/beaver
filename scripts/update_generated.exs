Mix.install([{:zig_parser, "~> 0.1.0"}])

defmodule Updater do
  def run do
    translate_out =
      IO.stream(:stdio, :line)
      |> Stream.map(&String.trim/1)
      |> Enum.join()

    zig_ast =
      Zig.Parser.parse(translate_out).code

    for {:fn, %Zig.Parser.FnOptions{extern: true, inline: inline}, parts} = f <-
          zig_ast,
        inline != true do
      {parts[:name], parts[:type], parts[:params] |> length()}
    end
    |> dbg
  end
end

Updater.run()
