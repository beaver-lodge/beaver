defmodule Beaver.Pattern.Native.Program do
  @moduledoc "A callback-free rewrite and dialect-conversion program executed natively in C++."
  alias Beaver.MLIR
  import Bitwise

  @enforce_keys [:root, :checks, :edits]
  defstruct [:root, :checks, :edits, benefit: 1]
  @type t :: %__MODULE__{root: String.t(), checks: list(), edits: list(), benefit: non_neg_integer()}

  def new(root, opts) when is_binary(root) and byte_size(root) > 0 do
    allowed = [:operands, :results, :attributes, :operand_types, :result_types, :same_operands, :rewrite, :benefit]

    unless Keyword.keyword?(opts) and Keyword.keys(opts) -- allowed == [] and
             length(Keyword.keys(opts)) == length(Enum.uniq(Keyword.keys(opts))),
           do: raise(ArgumentError, "invalid declarative pattern options")

    checks = Enum.flat_map(opts, &check/1)
    edits = edit(Keyword.fetch!(opts, :rewrite))
    benefit = Keyword.get(opts, :benefit, 1)
    unless benefit in 0..65_534, do: raise(ArgumentError, "invalid pattern benefit")
    %__MODULE__{root: root, checks: checks, edits: edits, benefit: benefit}
  end

  @doc "Creates a callback-free C ABI call lowering program for `root`."
  @spec c_abi_call(String.t(), String.t(), keyword()) :: t()
  def c_abi_call(root, target_op \\ "llvm.call", opts \\ []) when is_binary(root) and is_binary(target_op) do
    {pattern_opts, call_opts} =
      Keyword.split(opts, [:operands, :results, :attributes, :operand_types, :result_types, :same_operands, :benefit])

    new(root, Keyword.put(pattern_opts, :rewrite, {:c_abi_call, target_op, call_opts}))
  end

  defp check({kind, n}) when kind in [:operands, :results],
    do: [{if(kind == :operands, do: 0, else: 1), index!(n), nil, nil}]

  defp check({:attributes, attrs}),
    do: Enum.map(attrs, fn {name, value} -> {2, 0, to_string(name), attr_value!(value)} end)

  defp check({kind, types}) when kind in [:operand_types, :result_types],
    do: type_checks(if(kind == :operand_types, do: 3, else: 4), types)

  defp check({:same_operands, pairs}), do: Enum.map(pairs, fn {a, b} -> {5, index!(a), nil, index!(b)} end)

  defp check(_), do: []

  defp edit({:attributes, [_ | _] = attrs}) do
    names = Enum.map(attrs, fn {name, _} -> to_string(name) end)
    unless length(names) == length(Enum.uniq(names)), do: raise(ArgumentError, "duplicate rewrite attribute")
    Enum.map(attrs, fn {name, value} -> {0, 0, to_string(name), attr_value!(value)} end)
  end

  defp edit({:operand, index}), do: [{1, index!(index), nil, nil}]
  defp edit(:erase), do: [{2, 0, nil, nil}]
  defp edit({:replace_op, target_op}) when is_binary(target_op), do: edit({:replace_op, target_op, []})

  defp edit({:replace_op, target_op, opts}) when is_binary(target_op) and byte_size(target_op) > 0 and is_list(opts) do
    validate_keys!(
      opts,
      [:operands, :attributes, :copy_attributes, :result_types, :regions, :convert_function_type],
      ":replace_op"
    )

    {operand_mode, operand_edits} = encode_replace_operands!(Keyword.get(opts, :operands, :all))
    {copy_flag, copy_edits} = encode_copy_attrs!(Keyword.get(opts, :copy_attributes, false))

    flags =
      encode_regions_flag!(Keyword.get(opts, :regions, :none)) ||| copy_flag |||
        encode_convert_fn_flag!(Keyword.get(opts, :convert_function_type, false))

    attr_edits = Enum.map(Keyword.get(opts, :attributes, []), &encode_replace_attr!/1)
    result_edits = encode_replace_results!(Keyword.get(opts, :result_types, :preserve))
    [{3, operand_mode, target_op, flags}] ++ operand_edits ++ copy_edits ++ attr_edits ++ result_edits
  end

  defp edit({:c_abi_call, target_op, opts}) when is_binary(target_op) and byte_size(target_op) > 0 and is_list(opts) do
    validate_keys!(
      opts,
      [:params, :callee, :return_type, :stack_save, :ensure_declaration, :verify_declaration],
      ":c_abi_call"
    )

    params = Keyword.get(opts, :params, [])
    unless is_list(params), do: raise(ArgumentError, ":c_abi_call :params must be a list")
    stack_save = bool_flag(opts, :stack_save, true, 1)

    decl_flags = bool_flag(opts, :ensure_declaration, true, 1) ||| bool_flag(opts, :verify_declaration, false, 2)

    callee = encode_callee(Keyword.get(opts, :callee))
    {ret_type_ref, ret_type_edits} = encode_c_abi_return!(Keyword.get(opts, :return_type))
    param_edits = Enum.map(params, &encode_c_abi_param!/1)
    [{8, stack_save, target_op, {decl_flags, callee, ret_type_ref}}] ++ ret_type_edits ++ param_edits
  end

  defp edit(other), do: raise(ArgumentError, "unsupported declarative rewrite: #{inspect(other)}")

  defp validate_keys!(opts, allowed, label) do
    unless Keyword.keyword?(opts) and Keyword.keys(opts) -- allowed == [],
      do: raise(ArgumentError, "invalid #{label} options: #{inspect(opts)}")
  end

  defp bool_flag(opts, key, default, bit), do: if(Keyword.get(opts, key, default), do: bit, else: 0)

  defp encode_replace_operands!(:all), do: {-1, []}
  defp encode_replace_operands!(indices) when is_list(indices), do: {0, Enum.map(indices, &{4, index!(&1), nil, nil})}

  defp encode_replace_operands!(other),
    do: raise(ArgumentError, "invalid :replace_op :operands option: #{inspect(other)}")

  defp encode_regions_flag!(:none), do: 0
  defp encode_regions_flag!(:move), do: 1
  defp encode_regions_flag!(other), do: raise(ArgumentError, "invalid :replace_op :regions option: #{inspect(other)}")

  defp encode_copy_attrs!(false), do: {0, []}
  defp encode_copy_attrs!(copy_all) when copy_all in [true, :all], do: {2, []}

  defp encode_copy_attrs!({:except, names}) when is_list(names), do: {8, Enum.map(names, &{13, 0, to_string(&1), nil})}

  defp encode_copy_attrs!(names) when is_list(names), do: {0, Enum.map(names, &{6, 0, to_string(&1), nil})}

  defp encode_copy_attrs!(other),
    do: raise(ArgumentError, "invalid :replace_op :copy_attributes option: #{inspect(other)}")

  defp encode_convert_fn_flag!(false), do: 0
  defp encode_convert_fn_flag!(true), do: 4
  defp encode_convert_fn_flag!(mode) when mode in [:infer, :from_region], do: 4 ||| 16

  defp encode_convert_fn_flag!(other),
    do: raise(ArgumentError, "invalid :replace_op :convert_function_type option: #{inspect(other)}")

  defp encode_replace_attr!({name, {:symbol_ref, src_attr}})
       when (is_atom(src_attr) or is_binary(src_attr)) and src_attr != "" do
    {14, 0, to_string(name), to_string(src_attr)}
  end

  defp encode_replace_attr!({name, value}), do: {5, 0, to_string(name), attr_value!(value)}

  defp encode_replace_results!(:preserve), do: []
  defp encode_replace_results!([]), do: [{7, -1, nil, nil}]
  defp encode_replace_results!(types) when is_list(types), do: Enum.map(types, &{7, 0, nil, type_value!(&1)})

  defp encode_replace_results!(other),
    do: raise(ArgumentError, "invalid :replace_op :result_types option: #{inspect(other)}")

  defp encode_callee(nil), do: nil
  defp encode_callee(other), do: to_string(other) |> String.trim_leading("@")

  defp encode_c_abi_return!(v) when v in [nil, :void, :none], do: {nil, []}
  defp encode_c_abi_return!(%MLIR.Type{ref: ref}), do: {ref, []}
  defp encode_c_abi_return!(s) when is_binary(s), do: {nil, [{12, 0, nil, s}]}
  defp encode_c_abi_return!(other), do: raise(ArgumentError, "invalid :c_abi_call :return_type: #{inspect(other)}")

  defp encode_c_abi_param!(:in), do: {9, 0, nil, nil}
  defp encode_c_abi_param!({:in, _type}), do: {9, 0, nil, nil}
  defp encode_c_abi_param!(dir) when dir in [:in_ptr, :aggregate], do: {10, 0, nil, nil}
  defp encode_c_abi_param!({dir, type}) when dir in [:in_ptr, :aggregate], do: {10, 0, nil, type_value!(type)}
  defp encode_c_abi_param!({dir, type}) when dir in [:out, :out_ptr], do: {11, 1, nil, type_value!(type)}

  defp encode_c_abi_param!({dir, type, param_opts}) when dir in [:out, :out_ptr] and is_list(param_opts),
    do: {11, bool_flag(param_opts, :zero_init, true, 1), nil, type_value!(type)}

  defp encode_c_abi_param!(other), do: raise(ArgumentError, "invalid :c_abi_call parameter spec: #{inspect(other)}")

  defp type_checks(kind, types),
    do: Enum.map(types, fn {index, type} -> {kind, index!(index), nil, type_value!(type)} end)

  defp attr_value!(%MLIR.Attribute{ref: ref}), do: ref
  defp attr_value!(s) when is_binary(s) and byte_size(s) > 0, do: s

  defp attr_value!(other),
    do: raise(ArgumentError, "expected MLIR.Attribute or non-empty string, got: #{inspect(other)}")

  defp type_value!(%MLIR.Type{ref: ref}), do: ref
  defp type_value!(s) when is_binary(s) and byte_size(s) > 0, do: s
  defp type_value!(other), do: raise(ArgumentError, "expected MLIR.Type or non-empty string, got: #{inspect(other)}")

  defp index!(n) when is_integer(n) and n >= 0, do: n
  defp index!(_), do: raise(ArgumentError, "pattern counts and indices must be non-negative integers")
end
