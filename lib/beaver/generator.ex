defmodule Beaver.ComposerGenerator do
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
      alias Beaver.Composer

      # We are calling C functions dynamically at compile time, so we need to make sure managed libraries get loaded.

      for fa <- CAPI.__info__(:functions) do
        with {name, 0} <- fa do
          is_transform = name |> Atom.to_string() |> String.starts_with?(prefix)

          if is_transform do
            pass = apply(CAPI, name, [])

            arg_name =
              pass
              |> CAPI.beaverPassGetArgument()
              |> MLIR.to_string()

            pass_name =
              pass
              |> CAPI.beaverPassGetName()
              |> MLIR.to_string()

            normalized_name = Beaver.ComposerGenerator.normalized_name(arg_name)

            doc = pass |> CAPI.beaverPassGetDescription() |> MLIR.to_string()

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

            def unquote(normalized_name)(composer_or_op) do
              pass = CAPI.unquote(name)()
              Composer.append(composer_or_op, pass)
            end
          end
        end
      end
    end
  end
end
