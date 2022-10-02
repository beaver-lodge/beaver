defmodule Kinda.Prebuilt do
  defmacro __using__(opts) do
    force =
      quote do
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
end
