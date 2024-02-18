Mix.install([{:zig_parser, "~> 0.1.0"}])

defmodule Updater do
  def gen(functions, :elixir) do
    manifest = inspect(functions, pretty: true, limit: :infinity)
    File.write!("lib/beaver/mlir/capi_functions.exs", manifest)

    if mix_app_path = System.get_env("MIX_APP_PATH") do
      lib_name = System.get_env("KINDA_LIB_NAME")
      File.write!("#{mix_app_path}/native_install/kinda-meta-lib#{lib_name}.ex", manifest)
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
    return kinda.NIFFunc(Kinds, c_, name, .{ .flags = 1, .nif_name = nifPrefix ++ name });
    }
    const mlir_capi = @import("mlir_capi.zig");
    const K = mlir_capi.allKinds;
    pub const nif_entries = .{
    #{entries}
    };
    """

    dst = "native/mlir-zig-proj/src/wrapper.zig"
    File.write!(dst, txt)
    {_, 0} = System.cmd("zig", ["fmt", dst])
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
