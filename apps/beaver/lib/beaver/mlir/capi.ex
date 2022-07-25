defmodule Beaver.MLIR.CAPI do
  require Logger

  @moduledoc """
  This module calls C API of MLIR. These FFIs are generated from headers in LLVM repo and this repo's headers providing supplemental functions.
  """

  paths =
    Path.wildcard("native/mlir-c/**/*.h") ++
      Path.wildcard("native/mlir-c/**/*.cpp") ++
      Path.wildcard("native/mlir-zig/src/**") ++
      ["native/mlir-zig/#{Mix.env()}/build.zig"]

  paths = paths |> Enum.reject(&String.contains?(&1, "fizz.gen.zig"))

  for path <- paths do
    @external_resource path
  end

  wrapper_header_path = Path.join(File.cwd!(), "native/wrapper.h")
  beaver_include = Path.join(File.cwd!(), "native/mlir-c/include")

  %{
    nifs: nifs,
    resource_kinds: resource_kinds,
    dest_dir: dest_dir,
    zig_t_module_map: zig_t_module_map
  } =
    Fizz.gen(__MODULE__, wrapper_header_path, "native/mlir-zig",
      include_paths: %{
        llvm_include: Beaver.LLVM.Config.include_dir(),
        beaver_include: beaver_include
      },
      library_paths: %{
        beaver_libdir: Path.join([Mix.Project.build_path(), "native-install", "lib"])
      },
      type_gen: &__MODULE__.CodeGen.type_gen/2,
      nif_gen: &__MODULE__.CodeGen.nif_gen/1
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

    Logger.debug("[Beaver] loading NIF")

    with :ok <- :erlang.load_nif(nif_path, 0) do
      Logger.debug("[Beaver] NIF loaded")
      :ok
    else
      error -> error
    end
  end

  defmodule Ptr do
    defstruct ref: nil, zig_t: nil, bag: MapSet.new()
  end

  defmodule Array do
    defstruct ref: nil, zig_t: nil, bag: MapSet.new()
  end

  root_module = __MODULE__
  # generate resource modules
  for %Fizz.CodeGen.Type{
        module_name: module_name,
        zig_t: zig_t,
        delegates: delegates,
        fields: fields
      } = type <-
        resource_kinds,
      Atom.to_string(module_name)
      |> String.starts_with?(Atom.to_string(__MODULE__)) do
    Logger.debug("[Beaver] building resource module #{module_name}")

    defmodule module_name do
      @moduledoc """
      #{zig_t}
      """
      use Fizz.ResourceKind, root_module: root_module, zig_t: zig_t, fields: fields

      for {m, f, a} <- delegates do
        args = List.duplicate({:_, [if_undefined: :apply], Elixir}, a)

        defdelegate unquote(f)(unquote_splicing(args)), to: m
      end
    end
  end

  for f <- Path.wildcard("capi/src") do
    @external_resource f
  end

  # generate C function NIFs
  Logger.debug("[Beaver] generating NIF wrapper")
  @external_resource "capi/build.zig"
  for nif <- nifs do
    args_ast = Macro.generate_unique_arguments(nif.arity, __MODULE__)

    case nif do
      %Fizz.CodeGen.NIF{name: nil, nif_name: nif_name} ->
        @doc false
        def unquote(String.to_atom(nif_name))(unquote_splicing(args_ast)),
          do: "failed to load NIF"

      %Fizz.CodeGen.NIF{name: name, nif_name: nif_name, ret: ret} ->
        if ret == "void" do
          def unquote(String.to_atom(name))(unquote_splicing(args_ast)) do
            refs = Fizz.unwrap_ref([unquote_splicing(args_ast)])
            ref = apply(__MODULE__, unquote(String.to_atom(nif_name)), refs)
            :ok = check!(ref)
          end
        else
          return_module = Fizz.module_name(ret, __MODULE__, zig_t_module_map)

          def unquote(String.to_atom(name))(unquote_splicing(args_ast)) do
            refs = Fizz.unwrap_ref([unquote_splicing(args_ast)])
            ref = apply(__MODULE__, unquote(String.to_atom(nif_name)), refs)

            struct!(unquote(return_module), %{ref: check!(ref), zig_t: unquote(ret)})
          end
        end

        @doc false
        def unquote(String.to_atom(nif_name))(unquote_splicing(args_ast)),
          do: "failed to load MLIR NIF"
    end
  end

  require Fizz.CodeGen.Resource
  # generate resource NIFs
  for %{module_name: module_name} <- resource_kinds do
    Fizz.CodeGen.Resource.gen_resource_functions(module_name)
  end

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
  def beaver_raw_resource_bool_to_term(_), do: raise("NIF not loaded")
  def beaver_raw_resource_cstring_to_term_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_beaver_attribute_to_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_beaver_type_to_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_beaver_operation_to_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_get_resource_bool(_), do: raise("NIF not loaded")
  def beaver_raw_mlir_named_attribute_get(_, _), do: raise("NIF not loaded")
  def beaver_raw_get_resource_c_string(_), do: raise("NIF not loaded")

  def bool(value) when is_boolean(value),
    do: %__MODULE__.Bool{ref: beaver_raw_get_resource_bool(value)}

  # user APIs
  def check!(ref) do
    case ref do
      {:error, e} ->
        raise e

      ref ->
        ref
    end
  end

  # move this to Beaver.Data
  defmodule CString do
    defstruct ref: nil, zig_t: "[*c]const u8", bag: MapSet.new()
  end

  def c_string(value) when is_binary(value) do
    %__MODULE__.CString{ref: check!(beaver_raw_get_resource_c_string(value))}
  end

  def ptr(%{ref: ref, zig_t: zig_t}) do
    mod = Fizz.module_name(zig_t)
    maker = Module.concat([__MODULE__, mod, :ptr])

    struct!(__MODULE__.Ptr, %{
      ref: apply(__MODULE__, maker, [ref]) |> check!(),
      zig_t: Fizz.CodeGen.Type.ptr_type_name(zig_t)
    })
  end

  def opaque_ptr(%mod{ref: ref}) do
    maker = Module.concat([mod, :opaque_ptr])

    struct!(__MODULE__.Ptr, %{
      ref: apply(__MODULE__, maker, [ref]) |> check!(),
      zig_t: "?*anyopaque"
    })
  end

  def array(list, module, opts \\ [mut: false]) when is_list(list) do
    mut = Keyword.get(opts, :mut) || false
    func = if mut, do: "mut_array", else: "array"

    array_t =
      if mut do
        Fizz.CodeGen.Type.ptr_type_name(module.zig_t)
      else
        Fizz.CodeGen.Type.array_type_name(module.zig_t)
      end

    ref =
      apply(Beaver.MLIR.CAPI, Module.concat([module, func]), [
        Enum.map(list, &Fizz.unwrap_ref/1)
      ])
      |> Beaver.MLIR.CAPI.check!()

    %Beaver.MLIR.CAPI.Array{ref: ref, zig_t: array_t}
  end

  def to_term(%{ref: ref, zig_t: zig_t}) do
    mod = Fizz.module_name(zig_t)
    maker = Module.concat([__MODULE__, mod, :primitive])

    apply(
      __MODULE__,
      maker,
      [ref]
    )
    |> check!()
  end

  def bag(%{bag: bag} = v, list) when is_list(list) do
    %{v | bag: MapSet.union(MapSet.new(list), bag)}
  end

  def bag(%{bag: bag} = v, item) do
    %{v | bag: MapSet.put(bag, item)}
  end
end
