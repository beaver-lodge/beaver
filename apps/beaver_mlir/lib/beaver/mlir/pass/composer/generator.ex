defmodule Beaver.MLIR.Pass.Composer.Generator do
  @moduledoc false

  defmacro __using__(prefix: prefix) do
    quote bind_quoted: [prefix: prefix] do
      alias Beaver.MLIR
      alias Beaver.MLIR.CAPI
      alias Beaver.MLIR.Pass.Composer

      # We are calling C functions dynamically at compile time, so we need to make sure managed libraries get loaded.
      Application.ensure_all_started(:beaver_mlir)

      for fa <- CAPI.__info__(:functions) do
        with {name, 0} <- fa do
          is_transform = name |> Atom.to_string() |> String.contains?(prefix)

          if is_transform do
            pass = apply(CAPI, name, [])

            pass_arg =
              pass
              |> MLIR.Pass.normalized_name()

            doc = pass |> CAPI.beaverPassGetDescription() |> MLIR.StringRef.extract()

            @doc """
            #{doc}
            """
            def unquote(pass_arg)() do
              CAPI.unquote(name)()
            end

            def unquote(pass_arg)(composer_or_op = %Composer{}) do
              pass = CAPI.unquote(name)()
              Composer.add(composer_or_op, pass)
            end

            def unquote(pass_arg)(composer_or_op) do
              composer = %Composer{op: composer_or_op, passes: []}
              unquote(pass_arg)(composer)
            end
          end
        end
      end
    end
  end
end
