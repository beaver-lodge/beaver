defmodule Beaver.MLIR.CABI do
  @moduledoc "C ABI parameter conventions shared by native call lowering and value-wrapper templates."
  alias Beaver.{MLIR, MLIR.Template}

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
  def wrapper_source(name, callee, params, opts \\ []) do
    name = Template.check_symbol!(name)
    callee = Template.check_symbol!(callee)
    indexed = Enum.with_index(Enum.map(params, &parameter!/1))
    {_, results} = value_signature(params, opts[:return_type])
    raw_results = return_types(opts[:return_type])
    raw_types = for {{kind, type, _}, _} <- indexed, do: if(kind == :in, do: type_string(type), else: "!llvm.ptr")
    inputs = for {{kind, type, _}, i} <- indexed, kind != :out, do: "%in#{i}: #{type_string(type)}"
    {setup, restore} = stack_scope(indexed, opts)

    IO.iodata_to_binary([
      "func.func private #{callee}(#{Enum.join(raw_types, ", ")}) -> #{return_type(raw_results)}\n\n",
      "func.func private #{name}(#{Enum.join(inputs, ", ")}) -> (#{Enum.join(results, ", ")}) {\n",
      setup,
      Enum.map(indexed, &allocate/1),
      call_source(callee, indexed, raw_types, raw_results),
      load_outputs(indexed),
      restore,
      return_source(indexed, raw_results, results),
      "}\n"
    ])
  end

  defp stack_scope(indexed, opts) do
    uses_stack? = Enum.any?(indexed, fn {{kind, _, _}, _} -> kind != :in end)
    save? = uses_stack? and Keyword.get(opts, :stack_save, true)

    setup = [
      if(save?, do: "  %sp = \"llvm.intr.stacksave\"() : () -> !llvm.ptr\n", else: ""),
      if(uses_stack?, do: "  %one = llvm.mlir.constant(1 : i64) : i64\n", else: "")
    ]

    restore = if save?, do: "  \"llvm.intr.stackrestore\"(%sp) : (!llvm.ptr) -> ()\n", else: ""
    {setup, restore}
  end

  defp allocate({{:in, _, _}, _}), do: ""

  defp allocate({{kind, type, opts}, i}) do
    type = type_string(type)
    "  #{pointer(kind, i)} = llvm.alloca %one x #{type} : (i64) -> !llvm.ptr\n" <> initialize(kind, type, i, opts)
  end

  defp initialize(:aggregate, type, i, _), do: "  llvm.store %in#{i}, %agg#{i} : #{type}, !llvm.ptr\n"

  defp initialize(:out, type, i, opts) do
    if opts[:zero_init],
      do: "  %outz#{i} = llvm.mlir.zero : #{type}\n  llvm.store %outz#{i}, %outp#{i} : #{type}, !llvm.ptr\n",
      else: ""
  end

  defp call_source(callee, indexed, types, results) do
    args = for {{kind, _, _}, i} <- indexed, do: if(kind == :in, do: "%in#{i}", else: pointer(kind, i))
    assignment = if results == [], do: "", else: "%st = "

    "  #{assignment}func.call #{callee}(#{Enum.join(args, ", ")}) : (#{Enum.join(types, ", ")}) -> #{return_type(results)}\n"
  end

  defp return_type([]), do: "()"
  defp return_type([type]), do: type

  defp load_outputs(indexed) do
    for {{:out, type, _}, i} <- indexed, do: "  %outv#{i} = llvm.load %outp#{i} : !llvm.ptr -> #{type_string(type)}\n"
  end

  defp return_source(indexed, raw_results, results) do
    values = if(raw_results == [], do: [], else: ["%st"]) ++ for({{:out, _, _}, i} <- indexed, do: "%outv#{i}")
    if values == [], do: "  return\n", else: "  return #{Enum.join(values, ", ")} : #{Enum.join(results, ", ")}\n"
  end

  defp pointer(:aggregate, i), do: "%agg#{i}"
  defp pointer(:out, i), do: "%outp#{i}"
  defp return_types(type) when type in [nil, :none, :void], do: []
  defp return_types(type), do: [type_string(type)]
  defp type_string(%MLIR.Type{} = type), do: MLIR.to_string(type)
  defp type_string(nil), do: raise(ArgumentError, "C ABI wrapper parameters require explicit types")
  defp type_string(:ptr), do: "!llvm.ptr"
  defp type_string(type), do: to_string(type)
end
