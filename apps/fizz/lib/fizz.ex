defmodule Fizz do
  alias Fizz.CodeGen.{Function, Type, Resource, NIF}
  require Logger

  @moduledoc """
  Documentation for `Fizz`.
  """

  defp gen_kind_name_from_module_name(%Type{module_name: module_name, kind_name: nil} = t) do
    %{t | kind_name: Module.split(module_name) |> List.last()}
  end

  defp gen_kind_name_from_module_name(t), do: t

  defp gen_nif_name_from_module_name(
         module_name,
         %NIF{wrapper_name: wrapper_name, nif_name: nil} = nif
       ) do
    %{nif | nif_name: Module.concat(module_name, wrapper_name)}
  end

  defp gen_nif_name_from_module_name(_module_name, f), do: f

  @doc """
  Generate Zig code from a header and build a Zig project to produce a NIF library
  """
  def gen(
        root_module,
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
    type_gen = Keyword.get(opts, :type_gen) || (&Type.default/2)
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

    resource_kinds =
      types
      |> Enum.reject(fn x -> String.starts_with?(x, "[*c]") end)
      |> Enum.reject(fn x -> x in ["void"] end)
      |> Enum.map(fn x ->
        # TODO: support skip
        {:ok, t} = type_gen.(root_module, x)
        t |> gen_kind_name_from_module_name
      end)

    resource_kind_map =
      resource_kinds
      |> Enum.map(fn %{zig_t: zig_t, kind_name: kind_name} -> {zig_t, kind_name} end)
      |> Map.new()

    zig_t_module_map =
      resource_kinds
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
              var arg#{i}: #{Resource.resource_type_struct(arg, resource_kind_map)}.T = #{Resource.resource_type_resource_kind(arg, resource_kind_map)}.fetch(env, args[#{i}])
              catch
              return beam.make_error_binary(env, "fail to fetch resource for argument ##{i + 1}, expected: " ++ @typeName(#{Resource.resource_type_struct(arg, resource_kind_map)}.T));
            """
          end

        body =
          if ret == "void" do
            """
            #{Enum.join(arg_vars, "")}
            c.#{name}(#{Enum.join(proxy_arg_uses, ", ")});
            return beam.make_ok(env);
            """
          else
            """
            #{Enum.join(arg_vars, "")}
            return #{Resource.resource_type_resource_kind(ret, resource_kind_map)}.make(env, c.#{name}(#{Enum.join(proxy_arg_uses, ", ")}))
            catch return beam.make_error_binary(env, "fail to make resource for: " ++ @typeName(#{Resource.resource_type_struct(ret, resource_kind_map)}.T));
            """
          end

        """
        fn #{name}(env: beam.env, _: c_int, #{if length(args) == 0, do: "_", else: "args"}: [*c] const beam.term) callconv(.C) beam.term {
          #{body}
        }
        """
      end
      |> Enum.join("\n")

    resource_kinds_str =
      resource_kinds
      |> Enum.map(&Type.gen_resource_kind/1)
      |> Enum.join()

    resource_kinds_str_open_str =
      resource_kinds
      |> Enum.map(&Resource.resource_open/1)
      |> Enum.join()

    source = resource_kinds_str <> source

    nifs =
      Enum.map(functions, nif_gen)
      |> Enum.map(&gen_nif_name_from_module_name(root_module, &1))
      |> Enum.concat(List.flatten(Enum.map(resource_kinds, &NIF.from_resource_kind/1)))

    source = """
    #{source}
    pub fn open_generated_resource_types(env: beam.env) void {
    #{resource_kinds_str_open_str}

    // TODO: reverse the alias here
    kinda.aliasKind(kinda.Internal.USize, USize);
    kinda.aliasKind(kinda.Internal.OpaquePtr, OpaquePtr);
    kinda.aliasKind(kinda.Internal.OpaqueArray, OpaqueArray);
    }
    pub export const generated_nifs = .{
      #{nifs |> Enum.map(&Fizz.CodeGen.NIF.gen/1) |> Enum.join("  ")}
    }
    ++ #{Enum.map(resource_kinds, fn %{kind_name: kind_name} -> "#{kind_name}.nifs" end) |> Enum.join(" ++ \n")};
    """

    source =
      """
      pub const c = @cImport({
        @cDefine("_NO_CRT_STDIO_INLINE", "1");
        @cInclude("#{wrapper}");
      });
      const beam = @import("beam.zig");
      const kinda = @import("kinda.zig");
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

    %{
      nifs: nifs,
      resource_kinds: resource_kinds,
      dest_dir: dest_dir,
      zig_t_module_map: zig_t_module_map
    }
  end

  defp is_array(%Type{zig_t: type}) do
    is_array(type)
  end

  defp is_array("[*c]const " <> _type) do
    true
  end

  defp is_array(type) when is_binary(type) do
    false
  end

  defp is_ptr("[*c]" <> _type) do
    true
  end

  defp is_ptr(type) when is_binary(type) do
    false
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

  def module_name(zig_t, forward_module, zig_t_module_map) do
    if is_array(zig_t) do
      forward_module |> Module.concat("Array")
    else
      if is_ptr(zig_t) do
        forward_module |> Module.concat("Ptr")
      else
        zig_t_module_map |> Map.fetch!(zig_t)
      end
    end
  end
end
