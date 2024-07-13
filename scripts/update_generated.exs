Mix.install([{:zig_parser, "~> 0.1.0"}])

defmodule Updater do
  def args() do
    System.argv() |> Enum.chunk_every(2)
  end

  def gen(functions, :elixir) do
    manifest = inspect(functions, pretty: true, limit: :infinity)

    for ["--elixir", dst] <- args() do
      File.write!(dst, manifest)
    end

    functions
  end

  def gen(functions, :zig) do
    entries =
      for {name, _arity} <- functions do
        lazy_fns = ~w{mlirPassManagerRunOnOp} |> Enum.map(&String.to_atom/1)

        if name in lazy_fns do
          ~s{L(K, c, "#{name}"),}
        else
          ~s{N(K, c, "#{name}"),}
        end
      end
      |> Enum.join("\n")

    txt = """
    pub const c = @import("prelude.zig");
    const kinda = @import("kinda");
    const e = @import("erl_nif");
    const nifPrefix = "Elixir.Beaver.MLIR.CAPI.";
    pub fn N(comptime Kinds: anytype, c_: anytype, comptime name: anytype) e.ErlNifFunc {
    return kinda.NIFFunc(Kinds, c_, name, .{ .nif_name = nifPrefix ++ name });
    }
    pub fn L(comptime Kinds: anytype, c_: anytype, comptime name: anytype) e.ErlNifFunc {
    return kinda.NIFFunc(Kinds, c_, name, .{ .flags = e.ERL_NIF_DIRTY_JOB_CPU_BOUND, .nif_name = nifPrefix ++ name });
    }
    const mlir_capi = @import("mlir_capi.zig");
    const K = mlir_capi.allKinds;
    pub const nif_entries = .{
    #{entries}
    };
    """

    for ["--zig", dst] <- args() do
      File.write!(dst, txt)
      {_, 0} = System.cmd("zig", ["fmt", dst])
    end

    functions
  end

  def run do
    %{code: zig_ast} =
      IO.stream(:stdio, :line)
      |> Stream.map(&String.trim/1)
      |> Enum.join()
      |> Zig.Parser.parse()

    for {:fn, %Zig.Parser.FnOptions{extern: true, inline: inline}, parts} <- zig_ast,
        inline != true do
      {parts[:name], parts[:params] |> Enum.reject(&(&1 == :...)) |> length()}
    end
    |> Enum.sort()
    |> gen(:elixir)
    |> gen(:zig)
  end
end

Updater.run()
