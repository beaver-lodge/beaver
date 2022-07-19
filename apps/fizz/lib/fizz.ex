defmodule Fizz do
  require Logger

  @moduledoc """
  Documentation for `Fizz`.
  """

  @doc """
  Generate Zig code from a header and build a Zig project to produce a NIF library
  """
  def gen(wrapper, project_dir, opts \\ [include_paths: %{}, library_paths: %{}]) do
    Logger.debug("[Fizz] generating Zig code for wrapper: #{wrapper}")
    include_paths = Keyword.get(opts, :include_paths, %{})
    library_paths = Keyword.get(opts, :library_paths, %{})
    cache_root = Path.join([Mix.Project.build_path(), "mlir-zig-build", "zig-cache"])

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
      with {out, 0} <-
             System.cmd(
               "zig",
               ["translate-c", wrapper, "--cache-dir", cache_root] ++ include_path_args
             ) do
        out
      else
        {_error, _} ->
          raise "fail to run zig translate-c"
      end

    functions =
      String.split(out, "\n")
      |> Enum.filter(fn x -> String.contains?(x, "mlir") || String.contains?(x, "beaver") end)
      |> Enum.filter(fn x -> String.contains?(x, "pub extern fn") end)

    # collecting functions with zig translate
    prints =
      for f <- functions do
        "pub extern fn " <> name = f
        name = name |> String.split("(") |> Enum.at(0)

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
             System.cmd("zig", ["run", dst, "--cache-dir", cache_root] ++ include_path_args,
               stderr_to_stdout: true
             ) do
        out
      else
        {_error, ret_code} ->
          raise "fail to run reflection, ret_code: #{ret_code}"
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

    extra = primitive_types() |> Enum.map(&Fizz.CodeGen.Function.ptr_type_name/1)

    types = Enum.sort(types ++ extra) |> Enum.uniq()

    array_types =
      for type <- types do
        Function.array_type_name(type)
      end

    ptr_types =
      for type <- types do
        Function.ptr_type_name(type)
      end

    # generate wrapper.inc.zig source
    source =
      for %Function{name: name, args: args, ret: ret} <- functions do
        proxy_arg_uses =
          for {_arg, i} <- Enum.with_index(args) do
            "arg#{i}"
          end

        arg_vars =
          for {arg, i} <- Enum.with_index(args) do
            """
              var arg#{i}: #{arg} = undefined;
              if (beam.fetch_resource(#{arg}, env, #{Function.resource_type_var(arg)}, args[#{i}])) |value| {
                arg#{i} = value;
              } else |_| {
                return beam.make_error_binary(env, "fail to fetch resource for argument ##{i + 1}, expected: #{arg}");
              }
            """
          end

        """
        export fn fizz_nif_#{name}(env: beam.env, _: c_int, #{if length(args) == 0, do: "_", else: "args"}: [*c] const beam.term) beam.term {
        #{Enum.join(arg_vars, "")}
          var ptr : ?*anyopaque = e.enif_alloc_resource(#{Function.resource_type_var(ret)}, @sizeOf(#{ret}));

          const RType = #{ret};
          var obj : *RType = undefined;

          if (ptr == null) {
            unreachable();
          } else {
            obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
            obj.* = c.#{name}(#{Enum.join(proxy_arg_uses, ", ")});
          }
          return e.enif_make_resource(env, ptr);
        }
        """
      end
      |> Enum.join("\n")

    # types to generate array and pointer makers for. only C abi compatible types are supported.
    element_types =
      for "c.struct_" <> _ = type <- types do
        type
      end

    element_types = element_types ++ Enum.reject(primitive_types(), fn t -> t == "void" end)

    primitive_makers =
      for type <- primitive_types() do
        """
        export fn #{Function.primitive_maker_name(type)}(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
          var arg0: #{type} = undefined;
          if (beam.fetch_resource(#{type}, env, #{Function.resource_type_var(type)}, args[0])) |value| {
            arg0 = value;
          } else |_| {
            return beam.make_error_binary(env, "fail to fetch resource, expected: #{type}");
          }
          return beam.make(#{type}, env, arg0) catch beam.make_error_binary(env, "fail create primitive from resource of #{type}");
        }
        export fn #{Function.resource_maker_name(type)}(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
          var ptr: ?*anyopaque = e.enif_alloc_resource(#{Function.resource_type_var(type)}, @sizeOf(#{type}));
          const RType = #{type};
          var obj: *RType = undefined;
          if (ptr == null) {
              unreachable();
          } else {
              obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
          }
          if (beam.get(RType, env, args[0])) |value| {
              obj.* = value;
              return e.enif_make_resource(env, ptr);
          } else |_| {
              return beam.make_error_binary(env, "launching nif");
          }
        }
        """
      end
      |> Enum.join("\n")

    makers =
      for type <- element_types do
        """
        export fn #{Function.array_maker_name(type)}(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
          const ElementType: type = #{Function.process_type(type)};
          const resource_type_element = #{Function.resource_type_var(type)};
          const resource_type_array = #{Function.resource_type_var(Function.array_type_name(type))};
          return beam.get_resource_array_from_list(ElementType, env, resource_type_element, resource_type_array, args[0]) catch beam.make_error_binary(env, "fail create array of #{type}");
        }

        export fn #{Function.ptr_maker_name(type)}(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
          const ElementType: type = #{Function.process_type(type)};
          const resource_type_element = #{Function.resource_type_var(type)};
          const resource_type_array = #{Function.resource_type_var(Function.ptr_type_name(type))};
          return beam.get_resource_ptr_from_term(ElementType, env, resource_type_element, resource_type_array, args[0]) catch beam.make_error_binary(env, "fail create ptr of #{type}");
        }
        """
      end
      |> Enum.join("\n")

    makers = makers <> primitive_makers
    resource_types = Enum.uniq(types ++ array_types ++ ptr_types)

    nifs =
      Enum.map(functions, &Fizz.CodeGen.NIF.from_function/1) ++
        Enum.map(element_types, &Fizz.CodeGen.NIF.array_maker/1) ++
        Enum.map(element_types, &Fizz.CodeGen.NIF.ptr_maker/1) ++
        Enum.map(primitive_types(), &Fizz.CodeGen.NIF.resource_maker/1) ++
        Enum.map(primitive_types(), &Fizz.CodeGen.NIF.primitive_maker/1)

    source = """
    pub const c = @cImport({
      @cDefine("_NO_CRT_STDIO_INLINE", "1");
      @cInclude("#{wrapper}");
    });
    const beam = @import("beam.zig");
    const e = @import("erl_nif.zig");
    #{resource_types |> Enum.map(&Function.resource_type_global/1) |> Enum.join("")}
    #{source}
    #{makers}
    pub export fn __destroy__(_: beam.env, _: ?*anyopaque) void {
    }
    pub fn open_generated_resource_types(env: beam.env) void {
    #{Enum.map(resource_types, &Function.resource_open/1) |> Enum.join("")}
    }
    pub export var generated_nifs = [_]e.ErlNifFunc{
      #{nifs |> Enum.map(&Fizz.CodeGen.NIF.gen/1) |> Enum.join("  ")}
    };
    """

    src_dir = Path.join(project_dir, "src")
    File.write!(Path.join(src_dir, "mlir.fizz.gen.zig"), source)

    # generate build.inc.zig source
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
    dest_dir = "#{Path.join(Mix.Project.build_path(), "native-install")}"

    erts_include =
      Path.join([
        List.to_string(:code.root_dir()),
        "erts-#{:erlang.system_info(:version)}",
        "include"
      ])

    build_source =
      build_source <>
        """
        pub const cache_root = "#{cache_root}";
        pub const erts_include = "#{erts_include}";
        """

    File.write!(Path.join(src_dir, "build.fizz.gen.zig"), build_source)

    if Mix.env() in [:test, :dev] do
      with {_, 0} <- System.cmd("zig", ["fmt", "."], cd: project_dir) do
        :ok
      else
        {_error, _} ->
          Logger.warn("fail to run zig fmt")
      end
    end

    with {_, 0} <- System.cmd("zig", ["build", "--prefix", dest_dir], cd: project_dir) do
      Logger.debug("[Fizz] Zig library installed to: #{dest_dir}")
      :ok
    else
      {_error, ret_code} ->
        raise "fail to run zig compiler, ret_code: #{ret_code}"
    end

    {nifs, types, dest_dir}
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

  def module_name("[*c]const " <> type) do
    "Array#{module_name(type)}" |> String.to_atom()
  end

  def module_name("[*c]" <> type) do
    "Ptr#{module_name(type)}" |> String.to_atom()
  end

  def module_name(type) do
    type |> String.capitalize() |> String.to_atom()
  end

  def is_array("[*c]const " <> _type) do
    true
  end

  def is_array(type) when is_binary(type) do
    false
  end

  def element_type("[*c]const " <> type) do
    type
  end

  def unwrap_ref(%{ref: ref}) do
    ref
  end

  def unwrap_ref(arguments) when is_list(arguments) do
    Enum.map(arguments, &unwrap_ref/1)
  end

  def unwrap_ref(term) do
    term
  end

  def primitive_types do
    [
      "bool",
      "c_int",
      "c_uint",
      "f32",
      "f64",
      "i16",
      "i32",
      "i64",
      "i8",
      "isize",
      "u16",
      "u32",
      "u64",
      "u8",
      "usize",
      "void"
    ]
  end

  def is_primitive("void"), do: false

  def is_primitive(type) do
    type in primitive_types()
  end
end
