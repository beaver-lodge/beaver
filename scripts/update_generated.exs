Mix.install([{:zig_parser, "~> 0.1.0"}])

defmodule Updater do
  def run do
    translate_out =
      IO.stream(:stdio, :line)
      |> Stream.map(&String.trim/1)
      |> Enum.join()

    zig_ast =
      Zig.Parser.parse(translate_out).code

    functions =
      for {:fn, %Zig.Parser.FnOptions{extern: true, inline: inline}, parts} <-
            zig_ast,
          inline != true do
        {parts[:name], parts[:params] |> Enum.reject(&(&1 == :...)) |> length()}
      end
      |> Enum.sort()

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
    const kl = @import("kinda_library.zig");
    const e = @import("erl_nif");
    const nifPrefix = "Elixir.Beaver.MLIR.CAPI.";
    pub fn N(comptime Kinds: anytype, c_: anytype, comptime name: anytype) e.ErlNifFunc {
    return kl.KindaNIF(Kinds, c_, name, .{ .overwrite = nifPrefix ++ name });
    }
    pub fn L(comptime Kinds: anytype, c_: anytype, comptime name: anytype) e.ErlNifFunc {
    return kl.KindaNIF(Kinds, c_, name, .{ .flags = 1, .overwrite = nifPrefix ++ name });
    }
    const mlir_capi = @import("mlir_capi.zig");
    const K = mlir_capi.allKinds;
    pub const nif_entries = .{
    #{entries}
    };
    """

    dst = "native/mlir-zig-proj/src/wrapper.zig"
    File.write(dst, txt)
    {_, 0} = System.cmd("zig", ["fmt", dst])

    manifest = inspect(functions, pretty: true, limit: :infinity)
    File.write("lib/beaver/mlir/capi_functions.exs", manifest)
  end
end

Updater.run()
