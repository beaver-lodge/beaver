defmodule Exotic.Library do
  @moduledoc """
  Add `use Exotic.Library` and @native to declare functions in a C library.
  - Why there is no argument name in the C function declaration?
    It is designed to make the C function declaration as simple as possible.
    If you want argument name or a more seamless experienceyou could declare a wrapper function to call the auto-generated FFI function.
  """
  alias Exotic.{Type, Function}
  @enforce_keys [:ref, :path, :functions, :id]
  # when created by macro, the id is the module name
  defstruct [:ref, :path, :functions, :id]
  # TODO: check if module is still necessary

  defmacro __using__(opts) do
    quote do
      __outer__ = __MODULE__

      @before_compile Exotic.Library
      Module.register_attribute(__MODULE__, :native, accumulate: false, persist: true)
      Module.register_attribute(__MODULE__, :before_compile, accumulate: false, persist: true)
      Module.register_attribute(__MODULE__, :native_definitions, accumulate: false, persist: true)
      Module.register_attribute(__MODULE__, :include, accumulate: true)
      Module.register_attribute(__MODULE__, :function_signature, accumulate: true, persist: true)

      def native_definitions() do
        __MODULE__.__info__(:attributes)
        |> Keyword.get(:native_definition)
      end

      def library_paths() do
        paths = Keyword.get(unquote(opts), :path, [])

        case paths do
          paths when is_list(paths) ->
            paths

          path when is_binary(path) ->
            [path]
        end
      end

      def native_definition(fun) do
        native_definitions() |> Map.get(fun)
      end

      def load(path) do
        Exotic.load(__MODULE__, path)
      end

      def load!(path) do
        with {:ok, lib} <- Exotic.load(__MODULE__, path) do
          lib
        else
          {:error, error} ->
            raise error
        end
      end

      def load() do
        require Logger

        library_paths()
        |> Enum.reduce_while({:error, "not path given"}, fn
          path, acc ->
            Logger.debug("try loading library #{path}")

            with {:ok, lib} <- Exotic.load(__MODULE__, path) do
              {:halt, {:ok, lib}}
            else
              {:error, error} ->
                {:cont, {:error, error}}
            end
        end)
      end

      def load!() do
        with {:ok, lib} <- load() do
          lib
        else
          {:error, error} ->
            raise error
        end
      end

      defoverridable load!: 0

      defmodule Managed do
        use Agent
        @outer __outer__
        @moduledoc false
        def start_link(_) do
          Agent.start_link(fn -> @outer.load!() end, name: __MODULE__)
        end

        def value do
          case Process.get(__MODULE__) do
            nil ->
              global = Agent.get(__MODULE__, & &1)
              Process.put(__MODULE__, global)
              global

            managed ->
              managed
          end
        end
      end
    end
  end

  def transform_definition(
        {:v1, :def, _meta, [{_meta_, arg_types, [], return_type}]},
        {name, _arity}
      ) do
    arg_types =
      for arg_t <- arg_types do
        %Type{t: arg_t}
      end

    return_type = %Type{t: return_type}
    %Function.Definition{arg_types: arg_types, return_type: return_type, name: name}
  end

  defmacro __before_compile__(env) do
    native = Module.get_attribute(env.module, :native, [])

    funcs =
      for fun <- native do
        definition =
          Module.get_definition(env.module, fun)
          |> transform_definition(fun)

        Module.delete_definition(env.module, fun)
        {fun, definition}
      end
      |> Map.new()

    helpes =
      for {name, arity} <- native do
        args =
          0..arity
          |> Enum.to_list()
          |> Enum.take(arity)
          # TODO: there should be better ways to do this
          |> Enum.map(fn n -> {:"arg_#{n}", [line: 0], nil} end)

        quote do
          def unquote(name)(unquote_splicing(args)) do
            Exotic.call!(
              __MODULE__,
              unquote(name),
              [unquote_splicing(args)]
            )
          end
        end
      end

    includes = Module.get_attribute(env.module, :include, [])

    # TODO: generate typespecs
    codegens =
      for include <- includes do
        %Exotic.Header{functions: functions, structs: structs, type_defs: type_defs} =
          Exotic.Header.parse(%Exotic.Header{include | module_name: env.module})

        type_defs =
          for %Exotic.CodeGen.TypeDef{name: name, ty: type} <- type_defs do
            base_name =
              Module.split(name)
              |> List.last()

            should_hide_doc = String.starts_with?(base_name, "__")

            module_doc =
              if should_hide_doc do
                quote do
                  @doc false
                  @moduledoc false
                end
              else
                pretty = inspect(Macro.escape(type), pretty: true) |> Macro.escape()

                quote do
                  @moduledoc """
                  Type alias of:
                  ```
                  #{unquote(pretty)}
                  ```
                  """
                end
              end

            quote do
              defmodule unquote(name) do
                unquote(module_doc)

                use Exotic.TypeDef, as: unquote(Macro.escape(type))
              end
            end
          end

        structs =
          for %Exotic.CodeGen.Struct{name: name, fields: fields} <- structs do
            should_hide_doc =
              name
              |> Atom.to_string()
              |> then(fn x ->
                String.contains?(x, "_opaque_pthread") or String.contains?(x, "__darwin_pthread")
              end)

            fields_doc = Macro.escape(inspect(fields, pretty: true))

            module_doc =
              if should_hide_doc do
                quote do
                  @doc false
                  @moduledoc false
                end
              else
                quote do
                  @moduledoc """
                  struct of type:
                  ```
                  #{unquote(fields_doc)}
                  ```
                  """
                end
              end

            quote do
              defmodule unquote(name) do
                unquote(module_doc)
                use Exotic.Type.Struct, fields: unquote(Macro.escape(fields))
              end
            end
          end

        funcs =
          for cf = %Exotic.CodeGen.Function{name: name, args: args} <- functions do
            args =
              args
              |> Enum.map(fn {name, _type} -> {name, [line: 0], nil} end)

            f = Exotic.Function.Definition.from_code_gen(cf)

            quote do
              Module.put_attribute(__MODULE__, :function_signature, unquote(Macro.escape(cf)))

              def unquote(name)(unquote_splicing(args)) do
                Exotic.call!(
                  __MODULE__,
                  unquote(Macro.escape(f)),
                  [unquote_splicing(args)]
                )
              end
            end
          end

        definitions =
          for f = %Exotic.CodeGen.Function{name: name, args: args} <- functions do
            {{name, length(args)}, Exotic.Function.Definition.from_code_gen(f)}
          end

        ast = type_defs ++ structs ++ funcs
        {ast, definitions}
      end

    {codegens, header_definitions} =
      codegens
      |> Enum.reduce({[], []}, fn {ast, definitions}, {acc_ast, acc_definitions} ->
        {[ast | acc_ast], definitions ++ acc_definitions}
      end)

    funcs =
      Map.new(header_definitions)
      |> Map.merge(funcs)

    require Logger

    if Enum.empty?(funcs), do: Logger.warn("no native functions found in module #{env.module}")

    quote do
      # use a module to save meta info of this library, so it could be accessible in @on_load
      defmodule __MODULE__.Meta do
        @moduledoc false
        def native_definitions() do
          unquote(Macro.escape(funcs))
        end
      end

      @native_definitions unquote(Macro.escape(funcs))
      unquote(helpes)
      unquote(codegens)
    end
  end
end
