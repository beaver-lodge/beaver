defmodule Beaver.MLIR.Conversion.Ex do
  @moduledoc """
  Conversion plan for lowering the `ex` dialect to `func`/`arith`/`scf`/`cf`.

  M1 lowering for the scalar subset of the `ex` dialect:

    * `ex.lit` -> `arith.constant`
    * `ex.add` -> `arith.addi`
    * `ex.call` -> `func.call`
    * `ex.return` -> `func.return`
    * `ex.func` -> `func.func` (body region moved, `ex.return` types converted)

  The `ex` term types (`!ex.dyn`/`!ex.bound`/`!ex.unbound`) convert to a scalar
  word type (`i64`) until the Zig term runtime lands.

  `plan/1` returns a reusable `Beaver.MLIR.Conversion.Plan`; run it with
  `Beaver.MLIR.Conversion.Plan.run/2` and continue with the standard
  `arith-to-llvm` and `func-to-llvm` passes.

  `ex.var`/`ex.bind` must be materialized to SSA first with
  `Beaver.MLIR.Dialect.Ex.MaterializeBoundVariables`.
  """

  alias Beaver.Changeset
  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.Walker

  @doc """
  Returns a conversion plan lowering the `ex` scalar subset to `func`/`arith`.
  """
  @spec plan(keyword()) :: Plan.t()
  def plan(opts \\ []) do
    Plan.new(
      Keyword.merge(
        [
          mode: :full,
          folding_mode: :after_patterns,
          build_materializations: true
        ],
        opts
      )
    )
    |> Plan.add_legal_dialect("builtin")
    |> Plan.add_legal_dialect("func")
    |> Plan.add_legal_dialect("arith")
    |> Plan.add_legal_dialect("cf")
    |> Plan.add_legal_dialect("scf")
    |> Plan.add_illegal_dialect("ex")
    |> Plan.add_conversion(&convert_type/1, version: "1.0")
    |> Plan.add_conversion_pattern("ex.lit", &convert_lit/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.add", &convert_add/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.call", &convert_call/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.return", &convert_return/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.var", &convert_var/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.func", &convert_func/3, version: "1.0")
  end

  @doc """
  Converts an `ex` term type to its scalar word representation.
  """
  @spec convert_type(MLIR.Type.t()) :: MLIR.Type.t()
  def convert_type(type) do
    case MLIR.to_string(type) do
      "!ex.dyn" -> scalar_word(type)
      "!ex.bound" -> scalar_word(type)
      "!ex.unbound" -> scalar_word(type)
      _ -> type
    end
  end

  defp scalar_word(type) do
    ctx = MLIR.context(type)
    MLIR.Type.integer(64, ctx: ctx)
  end

  defp convert_lit(operation, [], rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)
    [result] = operation |> Walker.results() |> Enum.to_list()
    result_type = result |> MLIR.Value.type() |> convert_type()

    constant =
      %Changeset{name: "arith.constant", context: context, location: location}
      |> Changeset.add_argument(value: required_attribute(operation, "value"))
      |> Changeset.add_result(result_type)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, constant)

    MLIR.ConversionPatternRewriter.replace_op(
      rewriter,
      operation,
      MLIR.Operation.result(constant, 0)
    )

    :ok
  end

  defp convert_add(operation, [left, right], rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)
    [result] = operation |> Walker.results() |> Enum.to_list()
    result_type = result |> MLIR.Value.type() |> convert_type()

    add =
      %Changeset{name: "arith.addi", context: context, location: location}
      |> Changeset.add_argument([left, right])
      |> Changeset.add_result(result_type)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, add)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, MLIR.Operation.result(add, 0))
    :ok
  end

  defp convert_call(operation, args, rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)

    callee =
      operation
      |> required_attribute("callee")
      |> MLIR.CAPI.mlirStringAttrGetValue()
      |> MLIR.to_string()

    arity =
      operation
      |> required_attribute("arity")
      |> MLIR.CAPI.mlirIntegerAttrGetValueInt()
      |> Beaver.Native.to_term()

    unless length(args) == arity do
      raise ArgumentError,
            "ex.call arity attribute #{arity} does not match #{length(args)} arguments"
    end

    [result] = operation |> Walker.results() |> Enum.to_list()
    result_type = result |> MLIR.Value.type() |> convert_type()

    call =
      %Changeset{name: "func.call", context: context, location: location}
      |> Changeset.add_argument(args)
      |> Changeset.add_argument(callee: MLIR.Attribute.flat_symbol_ref(callee, ctx: context))
      |> Changeset.add_result(result_type)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, call)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, MLIR.Operation.result(call, 0))
    :ok
  end

  defp convert_return(operation, operands, rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)

    return_op =
      %Changeset{name: "func.return", context: context, location: location}
      |> Changeset.add_argument(operands)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, return_op)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, return_op)
    :ok
  end

  defp convert_var(operation, [], _rewriter) do
    raise ArgumentError,
          "ex.var without ex.bind is unsupported: #{inspect(MLIR.to_string(operation))}"
  end

  defp convert_func(operation, [], rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)

    sym_name =
      operation
      |> required_attribute("sym_name")
      |> MLIR.CAPI.mlirStringAttrGetValue()
      |> MLIR.to_string()

    [body_region] = operation |> Walker.regions() |> Enum.to_list()
    [block] = body_region |> Walker.blocks() |> Enum.to_list()
    terminator = MLIR.CAPI.mlirBlockGetTerminator(block)

    if MLIR.null?(terminator),
      do: raise(ArgumentError, "ex.func body must end with a terminator (ex.return)")

    return_types =
      terminator
      |> Walker.operands()
      |> Enum.to_list()
      |> Enum.map(&(&1 |> MLIR.Value.type() |> convert_type()))

    function_type = MLIR.Type.function([], return_types)

    func =
      %Changeset{name: "func.func", context: context, location: location}
      |> Changeset.add_argument(sym_name: MLIR.Attribute.string(sym_name))
      |> Changeset.add_argument(function_type: function_type)
      |> Changeset.add_argument(MLIR.CAPI.mlirRegionCreate())
      |> MLIR.Operation.create()

    func_body = func |> Walker.regions() |> Enum.to_list() |> hd()
    MLIR.CAPI.mlirRegionTakeBody(func_body, body_region)

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, func)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, func)
    :ok
  end

  defp required_attribute(operation, name) do
    case operation |> Walker.attributes() |> then(& &1[name]) do
      nil ->
        raise ArgumentError,
              "#{MLIR.Operation.name(operation)} requires attribute #{inspect(name)}"

      attribute ->
        attribute
    end
  end
end
