defmodule Beaver.Build.Command do
  @moduledoc "File-oriented build commands with CLI options and a JSONL request/reply loop."
  defmacro __using__(opts) do
    quote do
      def main(["--serve" | rest]), do: serve!(unquote(__MODULE__).options(rest))

      def main([source, output | rest]),
        do: tap(:ok, fn _ -> build!(source, output, unquote(__MODULE__).options(rest)) end)

      def main(_), do: raise(ArgumentError, unquote(Keyword.fetch!(opts, :usage)))

      def serve!(opts \\ []),
        do: unquote(__MODULE__).serve(&build!/3, opts, unquote(Keyword.get(opts, :source_alias, "source")))
    end
  end

  def options(["--cache-dir", directory | rest]), do: [{:cache_dir, directory} | options(rest)]
  def options(["--jobs", count | rest]), do: [{:max_concurrency, String.to_integer(count)} | options(rest)]
  def options([_ | rest]), do: options(rest)
  def options([]), do: []

  def serve(build, opts, source_alias) do
    for line <- IO.stream(:stdio, :line),
        request = String.trim(line),
        request != "",
        do: IO.puts(reply(request, build, opts, source_alias))
  end

  defp reply(line, build, opts, source_alias) do
    request = JSON.decode!(line)
    opts = if request["backend"], do: Keyword.put(opts, :backend, request["backend"]), else: opts
    build.(request[source_alias] || Map.fetch!(request, "source"), Map.fetch!(request, "out"), opts)
    ~s({"status":"ok"})
  rescue
    e -> JSON.encode!(%{"status" => "error", "message" => Exception.message(e)})
  end
end
