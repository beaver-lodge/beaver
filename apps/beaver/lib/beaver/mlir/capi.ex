defmodule Beaver.MLIR.CAPI do
  @moduledoc """
  This module calls C API of MLIR. These FFIs are generated from headers in LLVM repo and this repo's headers providing supplemental functions.
  """

  paths =
    Path.wildcard("native/mlir-c/include/**/*.h") ++
      Path.wildcard("native/mlir-zig/src/**") ++
      ["native/mlir-zig/build.zig"]

  for path <- paths do
    @external_resource path
  end

  wrapper_header_path = Path.join(File.cwd!(), "native/wrapper.h")
  beaver_include = Path.join(File.cwd!(), "native/mlir-c/include")

  {functions, types, dest_dir} =
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

    :erlang.load_nif(nif_path, 0)
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
  for %Fizz.CodeGen.Function{name: name, args: args, ret: ret} <- functions do
    arity = length(args)

    args_ast = Macro.generate_unique_arguments(arity, __MODULE__)

    fizz_func_name = String.to_atom("fizz_nif_" <> name)

    return_module = Module.concat(__MODULE__, Fizz.module_name(ret))

    def unquote(String.to_atom(name))(unquote_splicing(args_ast)) do
      refs = Fizz.unwrap_ref([unquote_splicing(args_ast)])
      ref = apply(__MODULE__, unquote(fizz_func_name), refs)
      struct!(unquote(return_module), %{ref: ref, zig_t: String.to_atom(unquote(ret))})
    end

    def unquote(fizz_func_name)(unquote_splicing(args_ast)),
      do: "failed to load NIF"
  end

  def registered_ops(), do: raise("NIF not loaded")
  def cstring_to_charlist(_), do: raise("NIF not loaded")
  def get_resource_bool(_), do: raise("NIF not loaded")
  def get_resource_c_string(_), do: raise("NIF not loaded")
  def bool(value) when is_boolean(value), do: %__MODULE__.Bool{ref: get_resource_bool(value)}

  def c_string(value) when is_binary(value) do
    case get_resource_c_string(value) do
      {:error, e} ->
        raise e

      ref ->
        %__MODULE__.CString{ref: ref}
    end
  end
end
