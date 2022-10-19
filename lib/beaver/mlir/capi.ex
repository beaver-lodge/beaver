defmodule Beaver.MLIR.CAPI do
  require Logger

  dest_dir = Path.join([Mix.Project.app_path(), "native-install"])

  llvm_constants =
    with {:ok, include_dir} <- Beaver.LLVM.Config.include_dir() do
      %{
        llvm_include: include_dir
      }
    else
      _ ->
        %{}
    end

  use Kinda.Prebuilt,
    otp_app: :beaver,
    lib_name: "beaver",
    base_url:
      Application.compile_env(
        :beaver,
        :prebuilt_base_url,
        "https://github.com/beaver-project/beaver-prebuilt/releases/download/2022-10-15-0706"
      ),
    version: "0.2.12",
    wrapper: Path.join(File.cwd!(), "native/wrapper.h"),
    zig_src: "native/mlir-zig-src",
    zig_proj: "native/mlir-zig-proj",
    include_paths:
      %{
        beaver_include: Path.join(File.cwd!(), "native/mlir-c/include")
      }
      |> Map.merge(llvm_constants),
    constants: %{
      beaver_libdir: Path.join(dest_dir, "lib")
    },
    dest_dir: dest_dir,
    forward_module: Beaver.Native,
    codegen_module: Beaver.MLIR.CAPI.CodeGen

  @moduledoc """

  This module calls C API of MLIR. These FFIs are generated from headers in LLVM repo and this repo's headers providing supplemental functions.
  """

  llvm_headers =
    with {:ok, include_dir} <- Beaver.LLVM.Config.include_dir() do
      include_dir
      |> Path.join("*.h")
      |> Path.wildcard()
    else
      _ ->
        []
    end

  # setting up elixir re-compilation triggered by changes in external files
  for path <-
        llvm_headers ++
          Path.wildcard("native/mlir-c/**/*.h") ++
          Path.wildcard("native/mlir-c/**/*.cpp") ++
          Path.wildcard("native/mlir-zig/src/**") ++
          ["native/mlir-zig/#{Mix.env()}/build.zig"],
      not String.contains?(path, "kinda.gen.zig") do
    @external_resource path
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
  def beaver_raw_beaver_value_to_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_beaver_affine_map_to_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_beaver_location_to_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_mlir_named_attribute_get(_, _), do: raise("NIF not loaded")
  def beaver_raw_get_resource_c_string(_), do: raise("NIF not loaded")
  def beaver_raw_read_opaque_ptr(_, _), do: raise("NIF not loaded")
  def beaver_raw_own_opaque_ptr(_), do: raise("NIF not loaded")
  def beaver_raw_context_attach_diagnostic_handler(_), do: raise("NIF not loaded")
end
