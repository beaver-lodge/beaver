if Version.match?(System.version(), "< 1.18.0") do
  Mix.install([
    {:jason, "~> 1.4"}
  ])
end

defmodule Updater do
  @with_diagnostics ~w{mlirAttributeParseGet mlirOperationVerify mlirTypeParseGet mlirModuleCreateParse beaverModuleApplyPatternsAndFoldGreedily mlirExecutionEngineCreate}
                    |> Enum.map(&String.to_atom/1)
  @normal_and_dirty ~w{mlirExecutionEngineInvokePacked}
                    |> Enum.map(&String.to_atom/1)

  defp dirty_io(name), do: "#{name}_dirty_io" |> String.to_atom()
  defp dirty_cpu(name), do: "#{name}_dirty_cpu" |> String.to_atom()
  defp with_diagnostics(name), do: "#{name}WithDiagnostics" |> String.to_atom()

  defp write_file(dst, txt) do
    tmp_dir = System.get_env("MIX_APP_PATH") || System.tmp_dir()

    if tmp_dir do
      tmp_dir
    else
      "./tmp" |> Path.expand() |> tap(&File.mkdir_p!/1)
    end

    tmp = Path.join(tmp_dir, "tmp-#{System.pid()}--#{Path.basename(dst)}")
    File.touch!(tmp)

    try do
      File.write!(tmp, txt)
      File.rename!(tmp, dst)
    rescue
      e ->
        File.rm!(tmp)
        reraise e, __STACKTRACE__
    end
  end

  defp gen(functions, :elixir, opts) do
    for {name, params} <- functions do
      params = params |> Enum.map(&String.to_atom/1)

      cond do
        name in @with_diagnostics or String.ends_with?(Atom.to_string(name), "GetChecked") ->
          [{name, params}, {with_diagnostics(name), [:context | params]}]

        name in @normal_and_dirty ->
          [{name, params}, {dirty_io(name), params}, {dirty_cpu(name), params}]

        true ->
          {name, params}
      end
    end
    |> List.flatten()
    |> inspect(pretty: true, limit: :infinity, printable_limit: :infinity)
    |> then(fn content ->
      if opts[:elixir] do
        dst = Path.expand(opts[:elixir])
        write_file(dst, content)
      end
    end)

    functions
  end

  defp gen(functions, :zig, opts) do
    entries =
      for {name, _arity} <- functions do
        cond do
          name in @with_diagnostics or String.ends_with?(Atom.to_string(name), "GetChecked") ->
            [~s{N(K, c, "#{name}"),}, ~s{diagnostic.WithDiagnosticsNIF(K, c, "#{name}"),}]

          name in @normal_and_dirty ->
            [
              ~s{N(K, c, "#{name}"),},
              ~s{D_IO(K, c, "#{name}", "#{dirty_io(name)}"),},
              ~s{D_CPU(K, c, "#{name}", "#{dirty_cpu(name)}"),}
            ]

          true ->
            ~s{N(K, c, "#{name}"),}
        end
      end
      |> List.flatten()
      |> Enum.join("\n")

    txt = """
    pub const c = @import("prelude.zig");
    pub const diagnostic = @import("diagnostic.zig");
    const N = c.N;
    const K = c.K;
    const D_CPU = c.D_CPU;
    const D_IO = c.D_IO;
    pub const nif_entries = .{
    #{entries}
    };
    """

    if opts[:zig] do
      dst = Path.expand(opts[:zig])
      write_file(dst, txt)
      {_, 0} = System.cmd("zig", ["fmt", dst], stderr_to_stdout: true)
    end

    functions
  end

  defp traverse(node) when is_list(node) do
    Enum.flat_map(node, &traverse/1)
  end

  defp traverse(node) when is_map(node) do
    function_decl = process_function_decl(node)
    children = traverse(Map.get(node, "inner", []))

    if function_decl do
      [function_decl | children]
    else
      children
    end
  end

  defp process_function_decl(%{"kind" => "FunctionDecl", "name" => name, "inner" => inner}) do
    params =
      for {elem, index} <- Enum.with_index(inner), elem["kind"] == "ParmVarDecl" do
        Map.get(elem, "name", "param_#{index}")
      end

    {String.to_atom(name), params}
  end

  defp process_function_decl(_), do: nil

  defp parse(txt) do
    json_mod = if Version.match?(System.version(), "< 1.18.0"), do: Jason, else: JSON

    apply(json_mod, :decode!, [txt])
    |> traverse()
    |> Enum.sort()
  end

  def run do
    opts =
      OptionParser.parse!(System.argv(),
        strict: [
          elixir: :string,
          zig: :string
        ]
      )
      |> elem(0)

    functions =
      IO.stream(:stdio, 10000)
      |> Enum.into("")
      |> parse()

    gen(functions, :elixir, opts)
    gen(functions, :zig, opts)
  end
end

Updater.run()
