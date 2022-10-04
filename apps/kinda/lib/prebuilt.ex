defmodule Kinda.Prebuilt do
  defmacro __using__(opts) do
    force =
      quote do
        opts = unquote(opts)
        contents = Kinda.Prebuilt.__using__(__MODULE__, opts)
        Module.eval_quoted(__MODULE__, contents)
      end

    quote do
      require Logger

      opts = unquote(opts)

      otp_app = Keyword.fetch!(opts, :otp_app)

      opts =
        Keyword.put_new(
          opts,
          :force_build,
          Application.compile_env(:kinda, [:force_build, otp_app])
        )

      case RustlerPrecompiled.__using__(__MODULE__, opts) do
        {:force_build, only_rustler_opts} ->
          unquote(force)

        {:ok, config} ->
          @on_load :load_rustler_precompiled
          @rustler_precompiled_load_from config.load_from
          @rustler_precompiled_load_data config.load_data

          @doc false
          def load_rustler_precompiled do
            # Remove any old modules that may be loaded so we don't get
            # {:error, {:upgrade, 'Upgrade not supported by this NIF library.'}}
            :code.purge(__MODULE__)
            {otp_app, path} = @rustler_precompiled_load_from

            load_path =
              otp_app
              |> Application.app_dir(path)
              |> to_charlist()

            :erlang.load_nif(load_path, @rustler_precompiled_load_data)
          end

        {:error, precomp_error} when is_bitstring(precomp_error) ->
          precomp_error
          |> String.split("You can force the project to build from scratch with")
          |> List.first()
          |> String.trim()
          |> Kernel.<>("""
          You can force the project to build from scratch with:
              config :kinda, :force_build, #{otp_app}: true
          """)
          |> raise

        {:error, precomp_error} ->
          raise precomp_error
      end
    end
  end

  def __using__(root_module, opts) do
    require Logger
    wrapper = Keyword.fetch!(opts, :wrapper)
    zig_src = Keyword.fetch!(opts, :zig_src)

    %{
      nifs: nifs,
      resource_kinds: resource_kinds,
      dest_dir: dest_dir,
      zig_t_module_map: zig_t_module_map
    } = Kinda.gen(root_module, wrapper, zig_src, opts)

    forward_module = Beaver.Native
    # generate resource modules
    kind_ast =
      for %Kinda.CodeGen.Type{
            module_name: module_name,
            zig_t: zig_t,
            fields: fields
          } = type <-
            resource_kinds,
          Atom.to_string(module_name)
          |> String.starts_with?(Atom.to_string(root_module)) do
        Logger.debug("[Beaver] building resource kind #{module_name}")

        quote bind_quoted: [
                root_module: root_module,
                module_name: module_name,
                zig_t: zig_t,
                fields: fields,
                forward_module: forward_module
              ] do
          defmodule module_name do
            @moduledoc """
            #{zig_t}
            """

            use Kinda.ResourceKind,
              root_module: root_module,
              fields: fields,
              forward_module: forward_module
          end
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
          t <- [Complex.F32, U8, U16, U32, I8, I16, I32, I64, F32, F64] do
        %Kinda.CodeGen.Type{
          module_name: Module.concat([Beaver.Native, t, MemRef, rank]),
          kind_functions: Beaver.MLIR.CAPI.CodeGen.memref_kind_functions()
        }
      end

    extra_kind_nifs =
      ([
         %Kinda.CodeGen.Type{
           module_name: Beaver.Native.PtrOwner
         },
         %Kinda.CodeGen.Type{
           module_name: Beaver.Native.Complex.F32,
           kind_functions: Beaver.MLIR.CAPI.CodeGen.memref_kind_functions()
         }
       ] ++ mem_ref_descriptor_kinds)
      |> Enum.map(&Kinda.CodeGen.NIF.from_resource_kind/1)
      |> List.flatten()

    nif_ast =
      for nif <- nifs ++ extra_kind_nifs do
        args_ast = Macro.generate_unique_arguments(nif.arity, __MODULE__)

        %Kinda.CodeGen.NIF{wrapper_name: wrapper_name, nif_name: nif_name, ret: ret} = nif

        stub_ast =
          quote do
            @doc false
            def unquote(nif_name)(unquote_splicing(args_ast)),
              do:
                raise(
                  "NIF for resource kind is not implemented, or failed to load NIF library. Function: :\"#{unquote(nif_name)}\"/#{unquote(nif.arity)}"
                )
          end

        wrapper_ast =
          if wrapper_name do
            if ret == "void" do
              quote do
                def unquote(String.to_atom(wrapper_name))(unquote_splicing(args_ast)) do
                  refs = Kinda.unwrap_ref([unquote_splicing(args_ast)])
                  ref = apply(__MODULE__, unquote(nif_name), refs)
                  :ok = unquote(forward_module).check!(ref)
                end
              end
            else
              return_module = Kinda.module_name(ret, forward_module, zig_t_module_map)

              quote do
                def unquote(String.to_atom(wrapper_name))(unquote_splicing(args_ast)) do
                  refs = Kinda.unwrap_ref([unquote_splicing(args_ast)])
                  ref = apply(__MODULE__, unquote(nif_name), refs)

                  struct!(unquote(return_module),
                    ref: unquote(forward_module).check!(ref)
                  )
                end
              end
            end
          end

        [stub_ast, wrapper_ast]
      end
      |> List.flatten()

    load_ast =
      quote do
        @dest_dir unquote(dest_dir)
        def kinda_on_load do
          require Logger
          nif_path = Path.join(@dest_dir, "lib/libBeaverNIF")
          dylib = "#{nif_path}.dylib"
          so = "#{nif_path}.so"

          if File.exists?(dylib) do
            File.ln_s(dylib, so)
          end

          Logger.debug("[Beaver] loading NIF, path: #{nif_path}")

          with :ok <- :erlang.load_nif(nif_path, 0) do
            Logger.debug("[Beaver] NIF loaded, path: #{nif_path}")
            :ok
          else
            error -> error
          end
        end
      end

    kind_ast ++ nif_ast ++ [load_ast]
  end
end
