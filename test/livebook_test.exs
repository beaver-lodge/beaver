defmodule LivebookTest do
  @moduledoc """
  Drift tests for the Livebook guides under `guides/`.

  Every `.livemd` is executed headlessly in CI: Elixir code cells are extracted
  from the notebook, cells that only install dependencies with `Mix.install`
  are skipped (the test runs inside the beaver project itself), and the
  remaining cells are evaluated top-to-bottom. This keeps the notebooks from
  drifting away from the APIs they demonstrate.

  Notebooks that rely on interactive cells (e.g. `Kino`) or network access are
  not suitable for this test.
  """

  use ExUnit.Case, async: true

  import ExUnit.CaptureIO

  @guides Path.expand("../guides", __DIR__)

  # Notebooks that print output should assert their expected output here so
  # regressions in the printed IR or the demonstrated pipeline are caught.
  @expected_output %{
    "your-first-beaver-compiler.livemd" => [
      "func.func @some_func",
      "llvm.func @some_func"
    ]
  }

  test "every guide notebook executes top-to-bottom" do
    livemds = Path.wildcard(Path.join(@guides, "*.livemd"))
    assert livemds != [], "no .livemd guides found under #{@guides}"

    for livemd <- livemds do
      name = Path.basename(livemd)
      cells = extract_cells(livemd)
      assert cells != [], "#{name}: no Elixir code cells found"

      code =
        cells
        |> Enum.reject(&mix_install?/1)
        |> Enum.join("\n\n")

      assert code != "", "#{name}: only Mix.install cells, nothing to execute"

      output =
        capture_io(fn ->
          Code.eval_string(code)
        end)

      for expected <- Map.get(@expected_output, name, []) do
        assert output =~ expected,
               "#{name}: expected printed output to contain #{inspect(expected)}"
      end
    end
  end

  defp extract_cells(livemd) do
    Regex.scan(~r/```elixir[^\n]*\n(.*?)\n```/s, File.read!(livemd), capture: :all_but_first)
    |> Enum.map(&hd/1)
  end

  defp mix_install?(cell) do
    cell |> String.trim() |> String.starts_with?("Mix.install")
  end
end
