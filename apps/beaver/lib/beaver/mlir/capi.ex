defmodule Beaver.MLIR.CAPI do
  require Logger

  @moduledoc """
  This module calls C API of MLIR. These FFIs are generated from headers in LLVM repo and this repo's headers providing supplemental functions.
  """

  paths =
    Path.wildcard("native/mlir-c/include/**/*.h") ++
      Path.wildcard("native/mlir-zig/src/**") ++
      ["native/mlir-zig/build.zig"]

  paths = paths |> Enum.reject(&String.contains?(&1, "fizz.gen.zig"))

  for path <- paths do
    @external_resource path
  end

  wrapper_header_path = Path.join(File.cwd!(), "native/wrapper.h")
  beaver_include = Path.join(File.cwd!(), "native/mlir-c/include")

  {nifs, types, dest_dir} =
    Fizz.gen(wrapper_header_path, "native/mlir-zig",
      include_paths: %{
        llvm_include: Beaver.LLVM.Config.include_dir(),
        beaver_include: beaver_include
      },
      library_paths: %{
        beaver_libdir: Path.join([Mix.Project.build_path(), "mlir-c-install", "lib"])
      }
    )

  @on_load :load_nifs

  @dest_dir dest_dir
  def load_nifs do
    nif_path = Path.join(@dest_dir, "lib/libBeaverNIF")
    dylib = "#{nif_path}.dylib"
    so = "#{nif_path}.so"

    if File.exists?(dylib) do
      File.ln_s(dylib, so)
    end

    Logger.debug("[MLIR] loading NIF")

    with :ok <- :erlang.load_nif(nif_path, 0) do
      Logger.debug("[MLIR] NIF loaded")
      :ok
    else
      error -> error
    end
  end

  for type <- types do
    defmodule Module.concat(__MODULE__, Fizz.module_name(type)) do
      defstruct ref: nil, zig_t: String.to_atom(type)
    end
  end

  for f <- Path.wildcard("capi/src") do
    @external_resource f
  end

  @external_resource "capi/build.zig"
  for nif <- nifs do
    args_ast = Macro.generate_unique_arguments(nif.arity, __MODULE__)

    case nif do
      %Fizz.CodeGen.NIF{name: nil, nif_name: nif_name} ->
        def unquote(String.to_atom(nif_name))(unquote_splicing(args_ast)),
          do: "failed to load NIF"

      %Fizz.CodeGen.NIF{name: name, nif_name: nif_name, arity: arity, ret: ret} ->
        return_module = Module.concat(__MODULE__, Fizz.module_name(ret))

        def unquote(String.to_atom(name))(unquote_splicing(args_ast)) do
          refs = Fizz.unwrap_ref([unquote_splicing(args_ast)])
          ref = apply(__MODULE__, unquote(String.to_atom(nif_name)), refs)

          struct!(unquote(return_module), %{ref: check!(ref), zig_t: String.to_atom(unquote(ret))})
        end

        def unquote(String.to_atom(nif_name))(unquote_splicing(args_ast)),
          do: "failed to load NIF"
    end
  end

  def registered_ops(), do: raise("NIF not loaded")
  def resource_bool_to_term(_), do: raise("NIF not loaded")
  def resource_cstring_to_term_charlist(_), do: raise("NIF not loaded")
  def get_resource_bool(_), do: raise("NIF not loaded")
  def get_resource_c_string(_), do: raise("NIF not loaded")
  def bool(value) when is_boolean(value), do: %__MODULE__.Bool{ref: get_resource_bool(value)}

  def check!(ref) do
    case ref do
      {:error, e} ->
        raise e

      ref ->
        ref
    end
  end

  def c_string(value) when is_binary(value) do
    %__MODULE__.CString{ref: check!(get_resource_c_string(value))}
  end

  def array(list) when is_list(list) do
    uniq = Enum.map(list, fn %mod{} -> mod end) |> Enum.uniq()

    case uniq |> Enum.count() do
      1 ->
        [%{zig_t: zig_t} | _] = list

        maker =
          zig_t
          |> Atom.to_string()
          |> Fizz.CodeGen.Function.array_maker_name()
          |> String.to_atom()

        ref = apply(__MODULE__, maker, [Enum.map(list, &Fizz.unwrap_ref/1)])

        zig_t
        |> Atom.to_string()
        |> Fizz.CodeGen.Function.array_type_name()
        |> Fizz.module_name()
        |> then(fn m -> Module.concat(__MODULE__, m) end)
        |> struct!(%{ref: check!(ref)})

      0 ->
        raise "TODO: return a null ptr"

      _ ->
        raise "not a list of same type"
    end
  end

  def to_term(%{ref: ref, zig_t: zig_t}) do
    apply(
      __MODULE__,
      String.to_atom(Fizz.CodeGen.Function.primitive_maker_name(Atom.to_string(zig_t))),
      [ref]
    )
  end

  def ptr(%{ref: ref, zig_t: zig_t}) do
    maker =
      zig_t
      |> Atom.to_string()
      |> Fizz.CodeGen.Function.ptr_maker_name()
      |> String.to_atom()

    ref = apply(__MODULE__, maker, [ref])

    zig_t
    |> Atom.to_string()
    |> Fizz.CodeGen.Function.ptr_type_name()
    |> Fizz.module_name()
    |> then(fn m -> Module.concat(__MODULE__, m) end)
    |> struct!(%{ref: check!(ref)})
  end
end
