defmodule Updater do
  require Logger

  def args() do
    System.argv() |> Enum.chunk_every(2)
  end

  @io_only ~w{mlirPassManagerRunOnOp} |> Enum.map(&String.to_atom/1)
  @regular_io_cpu ~w{mlirExecutionEngineInvokePacked} |> Enum.map(&String.to_atom/1)

  defp dirty_io(name), do: "#{name}_dirty_io" |> String.to_atom()
  defp dirty_cpu(name), do: "#{name}_dirty_cpu" |> String.to_atom()

  def gen(functions, :elixir) do
    for {name, arity} <- functions do
      if name in @regular_io_cpu do
        [{name, arity}, {dirty_io(name), arity}, {dirty_cpu(name), arity}]
      else
        {name, arity}
      end
    end
    |> List.flatten()
    |> inspect(pretty: true, limit: :infinity)
    |> then(
      &for ["--elixir", dst] <- args() do
        dst = Path.expand(dst)
        File.write!(dst, &1)
        Logger.info("Updated #{dst}")
      end
    )

    functions
  end

  def gen(functions, :zig) do
    entries =
      for {name, _arity} <- functions do
        cond do
          name in @io_only ->
            ~s{D_CPU(K, c, "#{name}", null),}

          name in @regular_io_cpu ->
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
    const N = c.N;
    const K = c.K;
    const D_CPU = c.D_CPU;
    const D_IO = c.D_IO;
    pub const nif_entries = .{
    #{entries}
    };
    """

    for ["--zig", dst] <- args() do
      File.write!(dst, txt)
      {_, 0} = System.cmd("zig", ["fmt", dst])
      dst = Path.expand(dst)
      Logger.info("Updated #{dst}")
    end

    functions
  end

  defp parse(tokens, acc \\ [])

  defp parse([], acc) do
    Enum.reverse(acc)
  end

  defp parse(["(" | tail], acc) do
    {tail, sub_acc} = parse(tail, [])
    parse(tail, [sub_acc | acc])
  end

  defp parse([")" | tail], acc) do
    {tail, Enum.reverse(acc)}
  end

  defp parse([head | tail], acc) do
    parse(tail, [head | acc])
  end

  @deliminator "()\:\;\,"
  defp tokenize(decl) do
    Regex.scan(~r/[^\s#{@deliminator}]+|[#{@deliminator}]/, decl) |> List.flatten()
  end

  defp fa(decl) do
    ["pub", "extern", "fn", name, args | _] = tokenize(decl) |> parse()

    args
    |> Enum.chunk_by(&(&1 == ","))
    |> Enum.reject(&(&1 in [[","], ["..."]]))
    |> length()
    |> then(&{String.to_atom(name), &1})
  end

  def run do
    IO.stream(:stdio, :line)
    |> Stream.map(&String.trim/1)
    |> Stream.filter(&String.starts_with?(&1, ["pub extern fn beaver", "pub extern fn mlir"]))
    |> Stream.map(&fa/1)
    |> Enum.sort()
    |> gen(:elixir)
    |> gen(:zig)
  end
end

Updater.run()
