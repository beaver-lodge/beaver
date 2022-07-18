defmodule Fizz do
  require Logger

  @moduledoc """
  Documentation for `Fizz`.
  """

  @doc """
  Generate Zig code from a header and build a Zig project to produce a NIF library
  """
  def gen(wrapper, project_dir, opts \\ [include_paths: %{}, library_paths: %{}]) do
    Logger.debug("[FizZ] Generating Zig code for wrapper: #{wrapper}")
    include_paths = Keyword.get(opts, :include_paths, %{})
    library_paths = Keyword.get(opts, :library_paths, %{})

    if not is_map(include_paths) do
      raise "include_paths must be a map so that we could generate variables for build.zig. Got: #{inspect(include_paths)}"
    end

    if not is_map(library_paths) do
      raise "library_paths must be a map so that we could generate variables for build.zig. Got: #{inspect(library_paths)}"
    end

    include_path_args =
      for {_, path} <- include_paths do
        ["-I", path]
      end
      |> List.flatten()

    out =
      with {out, 0} <- System.cmd("zig", ["translate-c", wrapper] ++ include_path_args) do
        out
      else
        {_error, _} ->
          raise "fail to run zig translate-c"
      end

    functions =
      String.split(out, "\n")
      |> Enum.filter(fn x -> String.contains?(x, "mlir") || String.contains?(x, "beaver") end)

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

    out =
      with {out, 0} <-
             System.cmd("zig", ["run", dst] ++ include_path_args, stderr_to_stdout: true) do
        out
      else
        {_error, _} ->
          raise "fail to run zig compiler"
      end

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
              var arg#{i}: #{arg} = undefined;
              if (beam.fetch_resource(arg#{i}, env, #{Function.resource_type_var(arg)}, args[#{i}])) |value| {
                arg#{i} = value;
              } else |_| {
                return beam.make_error_binary(env, "fail to fetch resource for arg#{i}");
              }
            """
          end

        """
        fn #{name}Wrapper(ret: anytype, #{Enum.join(proxy_args, ", ")}) void {
          ret.* = c.#{name}(#{Enum.join(proxy_arg_uses, ", ")});
        }

        export fn fizz_nif_#{name}(env: beam.env, _: c_int, #{if length(args) == 0, do: "_", else: "args"}: [*c] const beam.term) beam.term {
        #{Enum.join(arg_vars, "")}
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
    pub const c = @cImport({
      @cDefine("_NO_CRT_STDIO_INLINE", "1");
      @cInclude("#{wrapper}");
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

    src_dir = Path.join(project_dir, "src")
    File.write!(Path.join(src_dir, "mlir.fizz.gen.zig"), source)

    build_source =
      for {name, path} <- include_paths do
        """
        pub const #{name} = "#{path}";
        """
      end
      |> Enum.join()

    build_source =
      for {name, path} <- library_paths do
        """
        pub const #{name} = "#{path}";
        """
      end
      |> Enum.join()
      |> Kernel.<>(build_source)

    # TODO: make this a arg
    dest_dir = "#{Path.join(Mix.Project.build_path(), "mlir-c-install")}"

    erts_include =
      Path.join([
        List.to_string(:code.root_dir()),
        "erts-#{:erlang.system_info(:version)}",
        "include"
      ])

    build_source =
      build_source <>
        """
        pub const cache_root = "#{Path.join([Mix.Project.build_path(), "mlir-zig-build", "zig-cache"])}";
        pub const erts_include = "#{erts_include}";
        """

    File.write!(Path.join(src_dir, "build.fizz.gen.zig"), build_source)

    with {_, 0} <- System.cmd("zig", ["build", "--prefix", dest_dir], cd: project_dir) do
      Logger.debug("[FizZ] Zig library installed to: #{dest_dir}")
      :ok
    else
      {_error, _} ->
        raise "fail to run zig compiler"
    end

    {functions, types, dest_dir}
  end

  @doc """
  Get Elixir module name from a Zig type
  """
  def module_name("c.struct_" <> struct_name) do
    struct_name |> String.to_atom()
  end

  def module_name("[*c]const u8") do
    CString
  end

  def module_name(type) do
    type |> String.capitalize() |> String.to_atom()
  end

  def unwrap_ref(%{ref: ref}) do
    ref
  end

  def unwrap_ref(arguments) when is_list(arguments) do
    Enum.map(arguments, &unwrap_ref/1)
  end
end
