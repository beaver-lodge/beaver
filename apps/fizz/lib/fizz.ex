defmodule Fizz do
  alias Fizz.CodeGen.{Function, Type, Resource, NIF}
  require Logger

  @moduledoc """
  Documentation for `Fizz`.
  """

  defp gen_type(type, cb) do
    with {:ok, t} <- cb.(type) do
      t
    else
      _ -> raise "failed to generate type: #{type}"
    end
  end

  @doc """
  Generate Zig code from a header and build a Zig project to produce a NIF library
  """
  def gen(
        wrapper,
        project_dir,
        opts \\ [include_paths: %{}, library_paths: %{}, type_gen: nil, nif_gen: nil]
      ) do
    project_dir = Path.join(File.cwd!(), project_dir)
    source_dir = Path.join(project_dir, "src")
    project_dir = Path.join(project_dir, Atom.to_string(Mix.env()))
    project_source_dir = Path.join(project_dir, "src")
    Logger.debug("[Fizz] generating Zig code for wrapper: #{wrapper}")
    include_paths = Keyword.get(opts, :include_paths, %{})
    library_paths = Keyword.get(opts, :library_paths, %{})
    type_gen = Keyword.get(opts, :type_gen) || (&Type.default/1)
    nif_gen = Keyword.get(opts, :nif_gen) || (&NIF.from_function/1)
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
            if (arg_type.arg_type) |t| {
              print("arg: {s}\\n", .{ @typeName(t) });
            } else {
              unreachable;
            }
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
        File.write!("#{dst}.out.txt", out)
        out
      else
        {error, ret_code} ->
          Logger.error("[Zig] #{error}")
          raise "fail to run reflection, ret_code: #{ret_code}"
      end

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

    File.write!("#{dst}.functions.ex", inspect(functions, pretty: true, limit: :infinity))

    types =
      Enum.map(functions, fn f -> [f.ret, f.args] end)
      |> List.flatten()

    extra = primitive_types() |> Enum.map(&Type.ptr_type_name/1)

    types = Enum.sort(types ++ extra) |> Enum.uniq()

    resource_structs =
      types
      |> Enum.reject(fn x -> String.starts_with?(x, "[*c]") end)
      |> Enum.reject(fn x -> x in ["void"] end)
      |> Enum.map(&gen_type(&1, type_gen))

    resource_struct_map =
      resource_structs
      |> Enum.map(fn %{zig_t: zig_t, module_name: module_name} -> {zig_t, module_name} end)
      |> Map.new()

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
              var arg#{i}: #{Resource.resource_type_struct(arg, resource_struct_map)}.T = #{Resource.resource_type_resource_struct(arg, resource_struct_map)}.fetch(env, args[#{i}])
              catch
              return beam.make_error_binary(env, "fail to fetch resource for argument ##{i + 1}, expected: #{arg}");
            """
          end

        if ret == "void" do
          """
          export fn fizz_nif_#{name}(env: beam.env, _: c_int, #{if length(args) == 0, do: "_", else: "args"}: [*c] const beam.term) beam.term {
          #{Enum.join(arg_vars, "")}
            c.#{name}(#{Enum.join(proxy_arg_uses, ", ")});
            return beam.make_ok(env);
          }
          """
        else
          """
          export fn fizz_nif_#{name}(env: beam.env, _: c_int, #{if length(args) == 0, do: "_", else: "args"}: [*c] const beam.term) beam.term {
          #{Enum.join(arg_vars, "")}
            return beam.make_resource(env, c.#{name}(#{Enum.join(proxy_arg_uses, ", ")}), #{Resource.resource_type_var(ret, resource_struct_map)})
            catch return beam.make_error_binary(env, "fail to make resource for #{ret}");
          }
          """
        end
      end
      |> Enum.join("\n")

    types = Enum.map(types, &gen_type(&1, type_gen))

    resource_structs_str =
      resource_structs
      |> Enum.map(&Type.gen_resource_struct/1)
      |> Enum.join()

    resource_structs_str_open_str =
      resource_structs
      |> Enum.map(&Resource.resource_open/1)
      |> Enum.join()

    source = resource_structs_str <> source

    nifs = Enum.map(functions, nif_gen)

    source = """
    #{source}
    pub export fn __destroy__(_: beam.env, _: ?*anyopaque) void {
    }
    pub fn open_generated_resource_types(env: beam.env) void {
    #{resource_structs_str_open_str}
    }
    pub export const generated_nifs = .{
      #{nifs |> Enum.map(&Fizz.CodeGen.NIF.gen/1) |> Enum.join("  ")}
    } ++ #{Enum.map(resource_structs, fn %{module_name: module_name} -> "#{module_name}.nifs" end) |> Enum.join(" ++ ")};
    """

    source =
      """
      pub const c = @cImport({
        @cDefine("_NO_CRT_STDIO_INLINE", "1");
        @cInclude("#{wrapper}");
      });
      const beam = @import("beam.zig");
      const e = @import("erl_nif.zig");
      pub const root_module = "Elixir.Beaver.MLIR.CAPI";
      """ <> source

    dst = Path.join(project_source_dir, "mlir.imp.zig")
    File.mkdir_p(project_source_dir)
    Logger.debug("[Fizz] writing source import to: #{dst}")
    File.write!(dst, source)

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

    dst = Path.join(project_dir, "build.imp.zig")
    Logger.debug("[Fizz] writing build import to: #{dst}")
    File.write!(dst, build_source)

    if Mix.env() in [:test, :dev] do
      with {_, 0} <- System.cmd("zig", ["fmt", "."], cd: project_dir) do
        :ok
      else
        {_error, _} ->
          Logger.warn("fail to run zig fmt")
      end
    end

    sym_src_dir = Path.join(project_dir, "src")
    File.mkdir_p(sym_src_dir)

    for zig_source <- Path.wildcard(Path.join(source_dir, "*.zig")) do
      zig_source_link = Path.join(sym_src_dir, Path.basename(zig_source))
      Logger.debug("[Fizz] sym linking source #{zig_source} => #{zig_source_link}")

      if File.exists?(zig_source_link) do
        File.rm(zig_source_link)
      end

      File.ln_s(zig_source, zig_source_link)
    end

    Logger.debug("[Fizz] building Zig project in: #{project_dir}")

    with {_, 0} <- System.cmd("zig", ["build", "--prefix", dest_dir], cd: project_dir) do
      Logger.debug("[Fizz] Zig library installed to: #{dest_dir}")
      :ok
    else
      {_error, ret_code} ->
        raise "fail to run zig compiler, ret_code: #{ret_code}"
    end

    {nifs, types, resource_structs, dest_dir}
  end

  @doc """
  Get Elixir module name from a Zig type
  """
  def module_name(%Type{module_name: module_name}) do
    module_name
  end

  def module_name("c.struct_" <> struct_name) do
    struct_name |> String.to_atom()
  end

  def module_name("isize") do
    :ISize
  end

  def module_name("usize") do
    :USize
  end

  def module_name("c_int") do
    :CInt
  end

  def module_name("c_uint") do
    :CUInt
  end

  def module_name("[*c]const u8") do
    :CString
  end

  def module_name("[*c]const " <> type) do
    "Array#{module_name(type)}" |> String.to_atom()
  end

  def module_name("[*c]" <> type) do
    "Ptr#{module_name(type)}" |> String.to_atom()
  end

  def module_name("?*anyopaque") do
    :OpaquePtr
  end

  def module_name("?*const anyopaque") do
    :OpaqueArray
  end

  def module_name("?fn(?*anyopaque) callconv(.C) ?*anyopaque") do
    :ExternalPassConstruct
  end

  def module_name(
        "?fn(c.struct_MlirContext, ?*anyopaque) callconv(.C) c.struct_MlirLogicalResult"
      ) do
    :ExternalPassInitialize
  end

  def module_name(
        "?fn(c.struct_MlirOperation, c.struct_MlirExternalPass, ?*anyopaque) callconv(.C) void"
      ) do
    :ExternalPassRun
  end

  def module_name("?fn(" <> _ = fn_name) do
    raise "need module name for function type: #{fn_name}"
  end

  def module_name(type) do
    if String.contains?(type, "_") do
      raise "need module name for type: #{type}"
    else
      type |> String.capitalize() |> String.to_atom()
    end
  end

  def is_array(%Type{zig_t: type}) do
    is_array(type)
  end

  def is_array("[*c]const " <> _type) do
    true
  end

  def is_array(type) when is_binary(type) do
    false
  end

  def element_type(%Type{zig_t: type}) do
    element_type(type)
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
      "usize"
    ]
  end

  def is_primitive("void"), do: false

  def is_primitive(type) when is_binary(type) do
    type in primitive_types()
  end

  def is_primitive(%Type{zig_t: type}) do
    is_primitive(type)
  end

  def is_function("?fn(" <> _), do: true

  def is_function(_), do: false
end
