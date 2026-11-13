defmodule Beaver.MLIR.CABI do
  @moduledoc "C ABI parameter conventions shared by native call lowering and value-wrapper templates."
  alias Beaver.MLIR
  alias MLIR.IRBatch.{Builder, Expressions}
  import Expressions
  @doc "Normalizes a parameter to `{direction, type, options}`; aggregate inputs are passed by address."
  def parameter!(:in), do: {:in, nil, []}
  def parameter!({:in, type}), do: {:in, type, []}
  def parameter!(kind) when kind in [:aggregate, :in_ptr], do: {:aggregate, nil, []}
  def parameter!({kind, type}) when kind in [:aggregate, :in_ptr], do: {:aggregate, type, []}
  def parameter!({kind, type}) when kind in [:out, :out_ptr], do: {:out, type, [zero_init: true]}

  def parameter!({kind, type, opts}) when kind in [:out, :out_ptr] and is_list(opts),
    do: {:out, type, Keyword.put_new(opts, :zero_init, true)}

  def parameter!(other), do: raise(ArgumentError, "invalid :c_abi_call parameter spec: #{inspect(other)}")
  @doc "Returns textual value-level input and result types, with the C return value preceding output parameters."
  def value_signature(params, return_type \\ nil) do
    params = Enum.map(params, &parameter!/1)
    inputs = for {kind, type, _} <- params, kind != :out, do: type_string(type)
    results = return_types(return_type) ++ for({:out, type, _} <- params, do: type_string(type))
    {inputs, results}
  end

  @doc """
  Generates a private func.func wrapper and raw declaration using standard LLVM operations.
  Parameters require explicit types. Options: `:return_type` (void), `:stack_save` (true).
  Output slots are zero-initialized unless their parameter specifies `zero_init: false`.
  """
  def wrapper_source(name, callee, params, opts \\ []), do: source([wrapper_spec(name, callee, params, opts)]) <> "\n"
  @doc "Builds a context-independent wrapper specification for inclusion in a larger IRBatch module."
  def wrapper_spec(name, callee, params, opts \\ []) do
    for symbol <- [name, callee], do: MLIR.Template.check_symbol!(symbol)
    name = name |> to_string() |> String.trim_leading("@")
    callee = callee |> to_string() |> String.trim_leading("@")
    indexed = Enum.with_index(Enum.map(params, &parameter!/1))
    {_, results} = value_signature(params, opts[:return_type])
    inputs = for {{kind, type, _}, i} <- indexed, kind != :out, do: {"in#{i}", type_string(type)}

    function(name, inputs, results, fn b ->
      save? = Keyword.get(opts, :stack_save, true) and Enum.any?(indexed, fn {{kind, _, _}, _} -> kind != :in end)
      {sp, b} = if save?, do: Builder.emit(b, val("llvm.intr.stacksave", [], :ptr)), else: {nil, b}
      {args, b} = Enum.map_reduce(indexed, b, &allocate/2)
      {result, b} = Builder.emit(b, call(callee, args, return_types(opts[:return_type])))

      outputs =
        for {{{:out, type, _}, _}, ptr} <- Enum.zip(indexed, args), do: val("llvm.load", [ptr], type_string(type))

      {outputs, b} = Builder.emit(b, outputs)
      b = if save?, do: Builder.operation(b, "llvm.intr.stackrestore", [sp], [], [], nil), else: b
      ret(b, List.wrap(result) ++ outputs)
    end)
  end

  defp allocate({{:in, _, _}, i}, b), do: {"in#{i}", b}

  defp allocate({{kind, type, opts}, i}, b) do
    type = type_string(type)
    one = val("llvm.mlir.constant", [], :i64, value: {:i64, 1})
    {ptr, b} = Builder.emit(b, val("llvm.alloca", [one], :ptr, elem_type: {:type, type}))
    value = if kind == :aggregate, do: "in#{i}", else: val("llvm.mlir.zero", [], type)
    {ptr, if(kind == :aggregate or opts[:zero_init], do: store(b, ptr, value), else: b)}
  end

  defp return_types(type) when type in [nil, :none, :void], do: []
  defp return_types(type), do: [type_string(type)]
  defp type_string(%MLIR.Type{} = type), do: MLIR.to_string(type)
  defp type_string(nil), do: raise(ArgumentError, "C ABI wrapper parameters require explicit types")
  defp type_string(:ptr), do: "!llvm.ptr"
  defp type_string(type), do: to_string(type)
end
