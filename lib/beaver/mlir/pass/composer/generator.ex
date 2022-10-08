defmodule Beaver.MLIR.Pass.Composer.Generator do
  @moduledoc false
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  def normalized_name(original) do
    original
    |> String.replace("-", "_")
    |> Macro.underscore()
    |> String.to_atom()
  end

  defmacro __using__(prefix: prefix) do
    quote bind_quoted: [prefix: prefix] do
      alias Beaver.MLIR
      alias Beaver.MLIR.CAPI
      alias Beaver.MLIR.Pass.Composer

      # We are calling C functions dynamically at compile time, so we need to make sure managed libraries get loaded.

      for fa <- CAPI.__info__(:functions) do
        with {name, 0} <- fa do
          is_transform = name |> Atom.to_string() |> String.starts_with?(prefix)

          if is_transform do
            pass = apply(CAPI, name, [])

            arg_name =
              pass
              |> CAPI.beaverPassGetArgument()
              |> MLIR.StringRef.extract()

            pass_name =
              pass
              |> CAPI.beaverPassGetName()
              |> MLIR.StringRef.extract()

            normalized_name = MLIR.Pass.Composer.Generator.normalized_name(arg_name)

            doc = pass |> CAPI.beaverPassGetDescription() |> MLIR.StringRef.extract()

            @doc """
            #{doc}
            ### Argument name in MLIR CLI
            `#{arg_name}`
            ### Pass name in TableGen
            `#{pass_name}`
            """
            def unquote(normalized_name)() do
              CAPI.unquote(name)()
            end

            def unquote(normalized_name)(composer_or_op = %Composer{}) do
              pass = CAPI.unquote(name)()
              Composer.add(composer_or_op, pass)
            end

            def unquote(normalized_name)(composer_or_op) do
              composer = %Composer{op: composer_or_op, passes: []}
              unquote(normalized_name)(composer)
            end
          end
        end
      end
    end
  end
end
