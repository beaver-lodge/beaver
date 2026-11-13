defmodule Beaver.Build.Command do
  @moduledoc """
  Build commands and a JSONL request/reply service. Each request returns `{"ok": report}` or
  `{"error": message}`; an optional `id` is echoed. Diagnostics belong on stderr.
  A resident service keeps its configured backend, while failures leave it available for the next request.
  """
  @switches [cache_dir: :string, native_cache_dir: :string, jobs: :integer, backend: :string, profile: :string]
  defmacro __using__(opts) do
    quote do
      def main(["--serve" | rest]),
        do: serve!(unquote(__MODULE__).options(rest, unquote(opts[:defaults] || [])))

      def main([source, output | rest]) do
        options = unquote(__MODULE__).options(rest, unquote(opts[:defaults] || []))
        report = build!(source, output, options)
        IO.puts(JSON.encode!(report))
        report
      end

      def main(_), do: raise(ArgumentError, unquote(Keyword.fetch!(opts, :usage)))

      def serve!(opts \\ []),
        do: unquote(__MODULE__).serve(&build!/3, opts, unquote(Keyword.get(opts, :source_alias, "source")))
    end
  end

  def options(args, defaults \\ []) do
    {opts, rest} = OptionParser.parse!(args, strict: @switches)
    if rest != [], do: raise(ArgumentError, "unexpected build arguments: #{inspect(rest)}")
    normalize(Keyword.merge(defaults, opts))
  end

  def serve(build, opts, source_alias, io \\ :stdio) do
    for line <- IO.stream(io, :line),
        request = String.trim(line),
        request != "",
        do: IO.puts(io, reply(request, build, opts, source_alias))

    :ok
  end

  @doc "Processes one request without changing process IO or environment."
  def reply(line, build, opts \\ [], source_alias \\ "source") do
    request = JSON.decode!(line)
    unless is_map(request), do: raise(ArgumentError, "expected a build request object")
    response = execute(request, build, opts, source_alias)
    response = if Map.has_key?(request, "id"), do: Map.put(response, "id", request["id"]), else: response
    JSON.encode!(response)
  rescue
    error -> JSON.encode!(%{error: Exception.message(error)})
  end

  defp execute(request, build, opts, source_alias) do
    source = request[source_alias] || Map.fetch!(request, "source")
    output = Map.fetch!(request, "out")
    unless is_binary(source) and is_binary(output), do: raise(ArgumentError, "source and out must be paths")
    backend = request["backend"]

    if backend && opts[:backend] && backend != to_string(opts[:backend]),
      do: raise(ArgumentError, "driver backend mismatch")

    request_opts = for {key, _} <- @switches, Map.has_key?(request, to_string(key)), do: {key, request[to_string(key)]}
    %{ok: build.(source, output, normalize(Keyword.merge(opts, request_opts)))}
  rescue
    error -> %{error: Exception.message(error)}
  end

  defp normalize(opts) do
    {jobs, opts} = Keyword.pop(opts, :jobs)
    opts = if jobs, do: Keyword.put(opts, :max_concurrency, jobs), else: opts

    if opts[:max_concurrency] && not (is_integer(opts[:max_concurrency]) and opts[:max_concurrency] > 0),
      do: raise(ArgumentError, "jobs must be a positive integer")

    if opts[:backend] && to_string(opts[:backend]) not in ~w(llvm interp),
      do: raise(ArgumentError, "unsupported backend")

    if opts[:profile] do
      profile = %{"dev" => :dev, "balanced" => :balanced, "throughput" => :throughput}[to_string(opts[:profile])]
      if is_nil(profile), do: raise(ArgumentError, "unsupported profile")
      Keyword.put(opts, :profile, profile)
    else
      opts
    end
  end
end
