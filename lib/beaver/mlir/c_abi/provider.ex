defmodule Beaver.MLIR.CABI.Provider do
  @moduledoc """
  Header-driven C function bindings with caller-supplied ABI type and parameter maps.
  The admitted prototypes are one-line declarations with scalar, named aggregate or pointer parameters.
  Aggregate arguments cross the adapter by address; output ownership and layout remain caller policy.
  """
  alias Beaver.MLIR.{CABI, ExecutionEngine, Type}

  defmacro __using__(opts) do
    quote do
      @cabi unquote(__MODULE__).read!(unquote(opts))
      unquote_splicing(
        for name <- ~w(functions argument_types function_type value_parameters value_signature value_function_type
                       raw_symbol value_symbol symbol adapter_source)a do
          arity = %{functions: 0, symbol: 2}[name] || 1
          args = Macro.generate_arguments(arity, __MODULE__)

          quote do
            def unquote(name)(unquote_splicing(args)),
              do: unquote(__MODULE__).unquote(name)(@cabi, unquote_splicing(args))
          end
        end
      )

      defdelegate validate_namespace!(ns), to: unquote(__MODULE__)

      def symbol_map(library, ns, functions \\ functions()),
        do: unquote(__MODULE__).symbol_map(@cabi, library, ns, functions)

      def bind_library!(jit, library, ns, functions \\ functions()),
        do:
          Beaver.MLIR.ExecutionEngine.bind_library_symbols!(
            jit,
            library,
            unquote(__MODULE__).bindings(@cabi, ns, functions)
          )
    end
  end

  def read!(opts) do
    spec = Map.new(opts)
    regex = Regex.compile!("^#{Regex.escape(spec.c_return)} #{Regex.escape(spec.c_prefix)}(\\w+)\\(([^;]+)\\);$", "m")
    parameter = ~r/^\s*(?<t>(?:const )?\w+)(?:\s*(?<p>\*+))?(?:\s*\w+)?\s*$/
    normalize = fn p -> then(Regex.named_captures(parameter, p), &String.trim("#{&1["t"]} #{&1["p"]}")) end

    signatures =
      Map.new(Regex.scan(regex, File.read!(spec.header), capture: :all_but_first), fn [name, args] ->
        {name, Enum.map(String.split(args, ","), normalize)}
      end)

    Map.put(spec, :signatures, signatures)
  end

  def functions(spec), do: spec.signatures |> Map.keys() |> Enum.sort()
  def argument_types(spec, name), do: Enum.map(Map.fetch!(spec.signatures, name), &type(spec, &1))
  def function_type(spec, name), do: function_type({argument_types(spec, name), [spec.return_type]})

  def value_parameters(spec, name),
    do: Enum.map(Map.fetch!(spec.signatures, name), &(spec.parameters[&1] || {:in, type(spec, &1)}))

  def value_signature(spec, name), do: CABI.value_signature(value_parameters(spec, name), spec.return_type)
  def value_function_type(spec, name), do: function_type(value_signature(spec, name))

  defp function_type({inputs, results}),
    do: Type.function(Enum.map(inputs, &Type.get/1), Enum.map(results, &Type.get/1))

  defp type(spec, name), do: Map.get(spec.types, name, "!llvm.ptr")

  for {name, key} <- [raw_symbol: :raw_prefix, value_symbol: :value_prefix] do
    def unquote(name)(%{signatures: signatures} = spec, name) when is_map_key(signatures, name) do
      spec[unquote(key)] <> name
    end
  end

  def symbol(%{signatures: signatures} = spec, name, ns) when is_map_key(signatures, name) do
    spec.namespace_prefix <> validate_namespace!(ns) <> "_" <> name
  end

  def validate_namespace!(ns),
    do: if(is_binary(ns) and ns =~ ~r/\A\w{1,128}\z/, do: ns, else: raise(ArgumentError, "runtime_namespace"))

  def bindings(spec, ns, functions) do
    ns = validate_namespace!(ns)
    Map.new(functions, &{symbol(spec, &1, ns), [symbol(spec, &1, ns), spec.c_prefix <> &1]})
  end

  def symbol_map(spec, library, ns, functions) do
    Map.new(bindings(spec, ns, functions), fn {name, candidates} ->
      case ExecutionEngine.lookup_library_symbol(library, candidates) do
        {:ok, ptr} -> {name, ptr}
        :error -> raise ArgumentError, "missing #{name}"
      end
    end)
  end

  def adapter_source(spec, ns) do
    ns = validate_namespace!(ns)
    spec.prelude <> Enum.map_join(functions(spec), "\n", &adapter(spec, &1, ns))
  end

  defp adapter(spec, name, ns) do
    params = Enum.with_index(spec.signatures[name])
    decl = Enum.map_join(params, ", ", fn {t, i} -> "#{if t in spec.aggregates, do: "const #{t} *", else: t} a#{i}" end)
    checks = for {t, i} <- params, t in spec.aggregates, into: "", do: "if (!a#{i}) return #{spec.invalid}; "
    args = Enum.map_join(params, ", ", fn {t, i} -> "#{if t in spec.aggregates, do: "*"}a#{i}" end)

    "#{spec.export} #{spec.c_return} #{symbol(spec, name, ns)}(#{decl}) { #{checks}return #{spec.c_prefix}#{name}(#{args}); }\n"
  end
end
