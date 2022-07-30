defmodule Beaver.MLIR.CAPI do
  require Logger

  @moduledoc """
  This module calls C API of MLIR. These FFIs are generated from headers in LLVM repo and this repo's headers providing supplemental functions.
  """

  # setting up elixir re-compilation triggered by changes in external files
  for path <-
        Path.wildcard("native/mlir-c/**/*.h") ++
          Path.wildcard("native/mlir-c/**/*.cpp") ++
          Path.wildcard("native/mlir-zig/src/**") ++
          Path.wildcard("capi/src") ++
          ["native/mlir-zig/#{Mix.env()}/build.zig"],
      not String.contains?(path, "fizz.gen.zig") do
    if File.exists?(path) do
      Logger.debug("[Beaver] adding to elixir compiler external resource: #{path}")
      @external_resource path
    else
      raise "file not found: #{path}"
    end
  end

  # generate Zig and compile it as a NIF library
  %{
    nifs: nifs,
    resource_kinds: resource_kinds,
    dest_dir: dest_dir,
    zig_t_module_map: zig_t_module_map
  } =
    Fizz.gen(__MODULE__, Path.join(File.cwd!(), "native/wrapper.h"), "native/mlir-zig",
      include_paths: %{
        llvm_include: Beaver.LLVM.Config.include_dir(),
        beaver_include: Path.join(File.cwd!(), "native/mlir-c/include")
      },
      library_paths: %{
        beaver_libdir: Path.join([Mix.Project.build_path(), "native-install", "lib"])
      },
      type_gen: &__MODULE__.CodeGen.type_gen/2,
      nif_gen: &__MODULE__.CodeGen.nif_gen/1
    )

  root_module = __MODULE__
  forward_module = Beaver.Native
  # generate resource modules
  for %Fizz.CodeGen.Type{
        module_name: module_name,
        zig_t: zig_t,
        fields: fields
      } = type <-
        resource_kinds,
      Atom.to_string(module_name)
      |> String.starts_with?(Atom.to_string(__MODULE__)) do
    Logger.debug("[Beaver] building resource kind #{module_name}")

    defmodule module_name do
      @moduledoc """
      #{zig_t}
      """

      use Fizz.ResourceKind,
        root_module: root_module,
        fields: fields,
        forward_module: forward_module
    end
  end

  # generate stubs for generated NIFs
  Logger.debug("[Beaver] generating NIF wrappers")

  mem_ref_descriptor_kinds =
    for rank <- [
          DescriptorUnranked,
          Descriptor1D,
          Descriptor2D,
          Descriptor3D,
          Descriptor4D,
          Descriptor5D,
          Descriptor6D,
          Descriptor7D,
          Descriptor8D,
          Descriptor9D
        ],
        t <- [Complex.F32, U8, I32, I64, F32, F64] do
      %Fizz.CodeGen.Type{
        module_name: Module.concat([Beaver.Native, t, MemRef, rank]),
        kind_functions: Beaver.MLIR.CAPI.CodeGen.memref_kind_functions()
      }
    end

  extra_kind_nifs =
    ([
       %Fizz.CodeGen.Type{
         module_name: Beaver.Native.Complex.F32,
         kind_functions: Beaver.MLIR.CAPI.CodeGen.memref_kind_functions()
       }
     ] ++ mem_ref_descriptor_kinds)
    |> Enum.map(&Fizz.CodeGen.NIF.from_resource_kind/1)
    |> List.flatten()

  for nif <- nifs ++ extra_kind_nifs do
    args_ast = Macro.generate_unique_arguments(nif.arity, __MODULE__)

    %Fizz.CodeGen.NIF{wrapper_name: wrapper_name, nif_name: nif_name, ret: ret} = nif
    @doc false
    def unquote(nif_name)(unquote_splicing(args_ast)),
      do:
        raise(
          "NIF for resource kind is not implemented, or failed to load NIF library. Function: :\"#{unquote(nif_name)}\"/#{unquote(nif.arity)}"
        )

    if wrapper_name do
      if ret == "void" do
        def unquote(String.to_atom(wrapper_name))(unquote_splicing(args_ast)) do
          refs = Fizz.unwrap_ref([unquote_splicing(args_ast)])
          ref = apply(__MODULE__, unquote(nif_name), refs)
          :ok = unquote(forward_module).check!(ref)
        end
      else
        return_module = Fizz.module_name(ret, Beaver.Native, zig_t_module_map)

        def unquote(String.to_atom(wrapper_name))(unquote_splicing(args_ast)) do
          refs = Fizz.unwrap_ref([unquote_splicing(args_ast)])
          ref = apply(__MODULE__, unquote(nif_name), refs)

          struct!(unquote(return_module),
            ref: unquote(forward_module).check!(ref)
          )
        end
      end
    end
  end

  # stubs for hand-written NIFs
  def beaver_raw_get_context_load_all_dialects(), do: raise("NIF not loaded")

  def beaver_raw_create_mlir_pass(
        _name,
        _argument,
        _description,
        _op_name,
        _handler
      ),
      do: raise("NIF not loaded")

  def beaver_raw_pass_token_signal(_), do: raise("NIF not loaded")
  def beaver_raw_registered_ops(), do: raise("NIF not loaded")
  def beaver_raw_registered_ops_of_dialect(_), do: raise("NIF not loaded")
  def beaver_raw_registered_dialects(), do: raise("NIF not loaded")
  def beaver_raw_resource_c_string_to_term_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_beaver_attribute_to_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_beaver_type_to_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_beaver_operation_to_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_mlir_named_attribute_get(_, _), do: raise("NIF not loaded")
  def beaver_raw_get_resource_c_string(_), do: raise("NIF not loaded")
  def beaver_raw_read_opaque_ptr(_, _), do: raise("NIF not loaded")

  # setup NIF loading
  @on_load :load_nifs

  @dest_dir dest_dir
  def load_nifs do
    nif_path = Path.join(@dest_dir, "lib/libBeaverNIF")
    dylib = "#{nif_path}.dylib"
    so = "#{nif_path}.so"

    if File.exists?(dylib) do
      File.ln_s(dylib, so)
    end

    Logger.debug("[Beaver] loading NIF")

    with :ok <- :erlang.load_nif(nif_path, 0) do
      Logger.debug("[Beaver] NIF loaded")
      :ok
    else
      error -> error
    end
  end
end
