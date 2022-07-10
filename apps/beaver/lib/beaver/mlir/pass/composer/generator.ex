defmodule Beaver.MLIR.Pass.Composer.Generator do
  @moduledoc false
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  def normalized_name(lib, pass) do
    Exotic.call!(lib, :beaverPassGetArgument, [pass])
    |> MLIR.StringRef.extract()
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
          is_transform = name |> Atom.to_string() |> String.contains?(prefix)

          lib = CAPI.load!()

          if is_transform do
            pass = Exotic.call!(lib, name, [])

            pass_arg = MLIR.Pass.Composer.Generator.normalized_name(lib, pass)

            doc = Exotic.call!(lib, :beaverPassGetDescription, [pass]) |> MLIR.StringRef.extract()

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
