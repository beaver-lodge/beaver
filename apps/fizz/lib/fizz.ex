defmodule Fizz do
  @moduledoc """
  Documentation for `Fizz`.
  """

  @doc """
  Generate Zig code from a header
  """
  def gen(wrapper, opts \\ [include_paths: []]) do
    include_paths = Keyword.get(opts, :include_paths, [])

    include_paths =
      for i <- include_paths do
        ["-I", i]
      end
      |> List.flatten()

    {out, 0} = System.cmd("zig", ["translate-c", wrapper] ++ include_paths)

    functions =
      String.split(out, "\n")
      |> Enum.filter(fn x -> String.contains?(x, "mlir") end)

      # |> Enum.map(&IO.inspect/1)
      |> Enum.filter(fn x -> String.contains?(x, "pub extern fn") end)

    prints =
      for f <- functions do
        "pub extern fn " <> name = f
        name = name |> String.split("(") |> Enum.at(0)
        # |> IO.inspect()

        """
        {
        @setEvalBranchQuota(10000);
        const func_type = @typeInfo(@TypeOf(c.#{name}));
        comptime var i = 0;
        if (func_type.Fn.calling_convention == .C) {
          print("func: #{name}\\n", .{});
          inline while (i < func_type.Fn.args.len) : (i += 1) {
            const arg_type = func_type.Fn.args[i];
            print("arg: {}\\n", .{ arg_type.arg_type });
          }
          print("ret: {}\\n", .{ func_type.Fn.return_type });
        }
        }
        """
      end

    source = """
    const expect = @import("std").testing.expect;
    const print = @import("std").debug.print;
    const c = @cImport({
        @cDefine("_NO_CRT_STDIO_INLINE", "1");
        @cInclude("#{wrapper}");
    });
    pub fn main() !void {
        #{Enum.join(prints, "\n")}
    }
    """

    File.mkdir("tmp")
    dst = "tmp/print_arity.zig"
    File.write!(dst, source)

    {out, 0} = System.cmd("zig", ["run", dst] ++ include_paths, stderr_to_stdout: true)

    alias Fizz.CodeGen.Function

    functions =
      out
      |> String.trim()
      |> String.split("\n")
      |> Enum.reduce([], fn
        "func: " <> func, acc ->
          [%Function{name: func} | acc]

        "arg: " <> arg, [%Function{args: args} = f | tail] ->
          [%{f | args: args ++ [Function.process_type(arg)]} | tail]

        "ret: " <> ret, [%Function{ret: nil} = f | tail] ->
          [%{f | ret: Function.process_type(ret)} | tail]
      end)

    types =
      Enum.map(functions, fn f -> [f.ret, f.args] end)
      |> List.flatten()
      |> Enum.uniq()
      |> Enum.sort()

    source =
      for %Function{name: name, args: args, ret: ret} <- functions do
        proxy_args =
          for {_arg, i} <- Enum.with_index(args) do
            "arg#{i}: anytype"
          end

        proxy_arg_uses =
          for {_arg, i} <- Enum.with_index(args) do
            "arg#{i}"
          end

        arg_vars =
          for {arg, i} <- Enum.with_index(args) do
            """
            var arg#{i}: #{arg} = undefined; arg#{i} = beam.fetch_resource(arg#{i}, env, #{Function.resource_type_var(arg)}, args[#{i}]);
            """
          end

        """
        fn #{name}Wrapper(ret: anytype, #{Enum.join(proxy_args, ", ")}) void {
          ret.* = c.#{name}(#{Enum.join(proxy_arg_uses, ", ")});
        }

        export fn fizz_nif_#{name}(env: beam.env, _: c_int, #{if length(args) == 0, do: "_", else: "args"}: [*c] const beam.term) beam.term {
          #{Enum.join(arg_vars, "  ")}
          var ptr : ?*anyopaque = e.enif_alloc_resource(#{Function.resource_type_var(ret)}, @sizeOf(#{ret}));

          const RType = #{ret};
          var obj : *RType = undefined;

          if (ptr == null) {
            unreachable();
          } else {
            obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
          }
          #{name}Wrapper(obj, #{Enum.join(proxy_arg_uses, ", ")});
          return e.enif_make_resource(env, ptr);
        }
        """
      end
      |> Enum.join("\n")

    source = """
    const c = @cImport({
      @cDefine("_NO_CRT_STDIO_INLINE", "1");
      @cInclude("llvm-15.h");
    });
    const beam = @import("beam.zig");
    const e = @import("erl_nif.zig");
    #{types |> Enum.map(&Function.resource_type_global/1) |> Enum.join("")}
    #{source}

    pub export fn __destroy__(_: beam.env, _: ?*anyopaque) void {
    }
    pub fn open_generated_resource_types(env: beam.env) void {
      #{Enum.map(types, &Function.resource_open/1) |> Enum.join("  ")}
    }
    pub export var generated_nifs = [_]e.ErlNifFunc{
      #{Enum.map(functions, &Function.nif_declaration/1) |> Enum.join("  ")}
    };
    """

    zig_lib_dir = "capi"
    File.write!(Path.join([zig_lib_dir, "src", "mlir.fizz.gen.zig"]), source)

    build_source = """
    pub var llvm_include = "#{Fizz.LLVM.Config.include_dir()}";
    """

    File.write!(Path.join([zig_lib_dir, "src", "build.fizz.gen.zig"]), build_source)

    with {_, 0} <- System.cmd("zig", ["build"], cd: zig_lib_dir) do
      :ok
    else
      {_error, _} ->
        raise "fail to run zig compiler"
    end

    {functions, types}
  end

  @doc """
  Get Elixir module name from a Zig type
  """
  def module_name("c.struct_" <> struct_name) do
    struct_name |> String.to_atom()
  end

  def module_name(type) do
    type |> String.capitalize() |> String.to_atom()
  end
end
