defmodule Beaver.Shadow.Wasm do
  @moduledoc """
  WebAssembly-backed evaluator for `Beaver.Shadow.Runner`.

  This evaluator closes the Shadow Wavefront loop on a wasm runtime:

  ```text
  payload source → package_binary! (wasm32-wasi) → node runtime → invoke entry → receipt metadata
  ```

  It implements the same evaluator contract as `Runner`'s default surrogate:
  `(resolved, candidate) -> {:ok, value, metadata} | {:error, reason}`.
  Measurements (durations, runtime facts) go into `metadata` and never into
  `Receipt.identity/1`.

  On machines without a wasm runtime the evaluator does not crash: it returns
  `{:error, {:wasm_unavailable, reason}}` so the experiment loop degrades to a
  recorded failure with a distinguishable category — mirroring
  `:cuda_unavailable` on the GPU evaluator.
  """

  alias Beaver.MLIR
  alias Beaver.Shadow.Runner
  alias Beaver.Wasm

  @doc """
  Runs one wasm experiment over all candidates of `schedule` against `source`.

  Options are forwarded to `Runner.run/3` plus:

    * `:entry` — wasm export to invoke (default `"main"`)
    * `:args` — argument list for the entry call (default `[]`)
    * `:runner` — wasm runtime executable (default `"node"`)
    * `:target` — wasm target triple (default `"wasm32-wasi"`)
    * `:nostdlib` — compile without libc/WASI imports (default `true`)
  """
  @spec run(binary() | MLIR.Module.t(), Runner.schedule(), keyword()) :: Runner.Run.t()
  def run(source, schedule, opts \\ []) do
    evaluator = &evaluate(&1, &2, opts)
    Runner.run(source, schedule, Keyword.put(opts, :evaluator, evaluator))
  end

  @doc """
  Evaluator implementation for `Runner`.

  `opts` must provide `:source` (the payload to package and run).
  """
  @spec evaluate(Runner.resolved(), %{index: non_neg_integer(), choices: map()}, keyword()) ::
          {:ok, term(), map()} | {:error, term()}
  def evaluate(resolved, candidate, opts) do
    source = Keyword.fetch!(opts, :source)
    runner = Keyword.get(opts, :runner, "node")

    case runner_available?(runner) do
      false ->
        {:error, {:wasm_unavailable, "no #{runner} runtime on PATH"}}

      true ->
        evaluate_with_runtime(source, resolved, candidate, runner, opts)
    end
  end

  defp evaluate_with_runtime(source, resolved, candidate, runner, opts) do
    context = MLIR.Context.create()
    owned_module? = not match?(%MLIR.Module{}, source)

    try do
      module = compile_source!(source, context)

      try do
        package_and_run(module, resolved, candidate, runner, opts)
      after
        if owned_module?, do: MLIR.Module.destroy(module)
      end
    after
      MLIR.Context.destroy(context)
    end
  end

  defp compile_source!(%MLIR.Module{} = module, _context), do: module

  defp compile_source!(source, context) when is_binary(source) do
    MLIR.Module.create!(source, ctx: context)
  end

  defp package_and_run(module, resolved, candidate, runner, opts) do
    entry = Keyword.get(opts, :entry, "main")
    args = Keyword.get(opts, :args, [])
    target = Keyword.get(opts, :target, "wasm32-wasi")
    nostdlib = Keyword.get(opts, :nostdlib, true)

    started = System.monotonic_time()

    binary =
      Wasm.package_binary!(
        module,
        target: target,
        entry: entry,
        nostdlib: nostdlib
      )

    {outcome, node_ms} = run_in_runtime(binary, entry, args, runner)
    duration = System.monotonic_time() - started

    case outcome do
      {:ok, output} ->
        {:ok, output,
         %{
           artifact: %{
             cache: :miss,
             lookup_key: resolved.digest,
             artifact_key: "wasm:#{target}:#{entry}:#{resolved.digest}"
           },
           trace: %{
             action_count: 1,
             tags: ["wasm.run"],
             candidate_index: candidate.index,
             schedule_digest: resolved.digest
           },
           runtime: runtime_facts(runner),
           imports: binary.imports,
           exports: binary.exports,
           durations: %{run_ms: node_ms, total_native: duration}
         }}

      {:error, reason} ->
        {:error, {:wasm_run_failure, reason}}
    end
  end

  defp run_in_runtime(binary, entry, args, runner) do
    path = tmp_wasm_path()
    File.write!(path, binary.bytes)

    try do
      script = runner_script(binary.imports, entry, args)
      {output, status} = System.cmd(runner, ["-e", script, path], stderr_to_stdout: true)

      if status == 0 do
        case parse_node_output(output) do
          {:ok, %{"out" => out, "ms" => ms}} -> {{:ok, out}, ms}
          {:ok, %{"error" => message}} -> {{:error, message}, nil}
          other -> {{:error, {:unexpected_output, other}}, nil}
        end
      else
        {{:error, String.slice(output, -1000, 1000)}, nil}
      end
    after
      File.rm(path)
    end
  end

  defp runner_script(imports, entry, args) do
    imports_json = group_imports(imports) |> JSON.encode!()
    entry_json = JSON.encode!(entry)
    args_json = JSON.encode!(args)

    """
    const fs = require('fs');
    const wasm = fs.readFileSync(process.argv[1]);
    const imports = #{imports_json};
    const host = {};
    for (const [mod, entries] of Object.entries(imports)) {
      host[mod] = host[mod] || {};
      for (const e of entries) host[mod][e.name] = (...a) => 0n;
    }
    WebAssembly.instantiate(wasm, host).then(({instance}) => {
      const start = process.hrtime.bigint();
      let out;
      try {
        out = String(instance.exports[#{entry_json}](...#{args_json}));
      } catch (e) {
        console.log(JSON.stringify({error: String(e.message || e)}));
        return;
      }
      const ms = Number(process.hrtime.bigint() - start) / 1e6;
      console.log(JSON.stringify({out, ms}));
    }).catch(e => {
      console.log(JSON.stringify({error: String(e.message || e)}));
    });
    """
  end

  defp group_imports(imports) do
    Enum.reduce(imports, %{}, fn %{module: module, name: name}, acc ->
      Map.update(acc, module, [%{name: name}], &(&1 ++ [%{name: name}]))
    end)
  end

  defp parse_node_output(output) do
    output
    |> String.split("\n", trim: true)
    |> Enum.reverse()
    |> Enum.find_value(fn line ->
      case JSON.decode(line) do
        {:ok, map} when is_map(map) -> {:ok, map}
        _ -> nil
      end
    end)
  end

  defp runner_available?(runner) do
    case Path.type(runner) do
      :absolute -> File.regular?(runner)
      _ -> System.find_executable(runner) != nil
    end
  end

  defp runtime_facts(runner) do
    {version, 0} = System.cmd(runner, ["--version"], stderr_to_stdout: true)
    %{runner: runner, version: String.trim(version)}
  rescue
    _ -> %{runner: runner, version: nil}
  end

  defp tmp_wasm_path do
    Path.join(System.tmp_dir!(), "shadow_wasm_#{System.unique_integer([:positive])}.wasm")
  end
end
