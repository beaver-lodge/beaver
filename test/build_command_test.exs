defmodule Beaver.BuildCommandTest do
  use ExUnit.Case, async: true
  alias Beaver.Build.Command

  test "the service returns complete reports, correlation IDs and explicit request options" do
    report = %{backend: "llvm", entries: [["entry/0", "entry/0", 0, 0]], compilation: [%{codegen_ns: 12}]}

    build = fn source, out, opts ->
      assert {source, out} == {"guest.json", "image"}
      assert opts[:backend] == "llvm"
      assert opts[:profile] == :dev
      assert opts[:max_concurrency] == 2
      report
    end

    request = JSON.encode!(%{id: 7, source: "guest.json", out: "image", backend: "llvm", profile: "dev", jobs: 2})

    assert Command.reply(request, build, backend: :llvm) |> JSON.decode!() ==
             JSON.decode!(JSON.encode!(%{id: 7, ok: report}))
  end

  test "malformed and failed requests do not stop subsequent requests or change the resident backend" do
    lines = [
      "{",
      "[]",
      ~s({"id":1,"source":"bad","out":"out"}),
      ~s({"id":2,"source":"valid","out":"out","backend":"interp"}),
      ~s({"id":3,"guest":"valid","out":"out","backend":"llvm"})
    ]

    {:ok, io} = StringIO.open(Enum.join(lines, "\n") <> "\n")

    build = fn
      "bad", _, _ -> raise "compile failed"
      "valid", "out", opts -> %{backend: opts[:backend], image: "out/program-image.json"}
    end

    assert :ok = Command.serve(build, [backend: "llvm"], "guest", io)
    {:ok, {_, output}} = StringIO.close(io)
    [invalid, object, failed, mismatch, success] = output |> String.split("\n", trim: true) |> Enum.map(&JSON.decode!/1)
    assert Map.has_key?(invalid, "error")
    assert object == %{"error" => "expected a build request object"}
    assert failed == %{"id" => 1, "error" => "compile failed"}
    assert mismatch == %{"id" => 2, "error" => "driver backend mismatch"}
    assert success == %{"id" => 3, "ok" => %{"backend" => "llvm", "image" => "out/program-image.json"}}
  end

  test "invalid options fail before building" do
    cases = [
      {["--jobs", "0"], ArgumentError},
      {["--profile", "unknown"], ArgumentError},
      {["--backend", "native"], ArgumentError},
      {["--typo", "x"], OptionParser.ParseError}
    ]

    for {args, error} <- cases, do: assert_raise(error, fn -> Command.options(args) end)
    assert Command.options(["--jobs", "3"], backend: "llvm") == [max_concurrency: 3, backend: "llvm"]
  end
end
