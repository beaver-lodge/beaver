defmodule Beaver.MLIR.Pattern do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Although this module is `MLIR.Pattern`, at this point it is a synonym of PDL patterns.
  Pattern-matching is done by MLIR which works in a different way from Erlang pattern-matching.
  The major difference is that MLIR pattern-matching will greedily match the patterns and maximize the benifit.
  Compiled patterns will be saved as module attributes in MLIR assembly format.
  """
  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__), only: [pattern: 2]
      Module.register_attribute(__MODULE__, :compiled_pattern, accumulate: true, persist: true)
    end
  end

  defmacro pattern(call, do: block) do
    ex_loc = Macro.Env.location(__ENV__)
    ctx = MLIR.Context.create()
    loc = MLIR.Location.get!(ctx, ex_loc)
    _module = MLIR.CAPI.mlirModuleCreateEmpty(loc)
    {name, args} = Macro.decompose_call(call)

    region = CAPI.mlirRegionCreate()
    pattern_block = MLIR.Block.create([], [])
    CAPI.mlirRegionAppendOwnedBlock(region, pattern_block)

    pdl_pattern_op =
      MLIR.Operation.State.get!("pdl.pattern", loc)
      |> MLIR.Operation.State.add_regions([region])
      |> MLIR.Operation.State.add_attr(sym_name: "\"#{name}\"", benefit: "1 : i16")
      |> MLIR.Operation.create()

    # create constraints in the matcher body of a `pdl.pattern` from arguments
    {_ast, pattern_code_gen_ctx} =
      args
      |> Macro.postwalk(
        %MLIR.Pattern.CodeGen.Context{block: pattern_block, mlir_ctx: ctx, ex_loc: ex_loc},
        &MLIR.Pattern.CodeGen.from_ast/2
      )

    # TODO: get root by querying value of first node in prewalk
    # create rewrite

    rewrite_region = CAPI.mlirRegionCreate()
    rewrite_block = MLIR.Block.create([], [])
    CAPI.mlirRegionAppendOwnedBlock(rewrite_region, rewrite_block)

    pdl_rewrite_op =
      MLIR.Operation.State.get!("pdl.rewrite", loc)
      |> MLIR.Operation.State.add_operand([pattern_code_gen_ctx.root])
      |> MLIR.Operation.State.add_regions([rewrite_region])
      |> MLIR.Operation.State.add_attr(operand_segment_sizes: "dense<[1, 0]> : vector<2xi32>")
      |> MLIR.Operation.create()

    # create ops in rewrite block

    {_ast, _rewrite_code_gen_ctx} =
      block
      |> Macro.postwalk(
        %MLIR.Pattern.CodeGen.Context{pattern_code_gen_ctx | block: rewrite_block},
        &MLIR.Pattern.CodeGen.from_ast/2
      )

    CAPI.mlirBlockAppendOwnedOperation(pattern_block, pdl_rewrite_op)

    pattern_string = MLIR.Operation.to_string(pdl_pattern_op) |> String.trim()

    MLIR.Operation.verify!(pdl_pattern_op)
    CAPI.mlirContextDestroy(ctx)

    quote do
      @compiled_pattern unquote(pattern_string)
    end
  end

  def compiled_patterns(module) when is_atom(module) do
    apply(module, :__info__, [:attributes])[:compiled_pattern]
  end

  def from_string(ctx, pdl_pattern_str) when is_binary(pdl_pattern_str) do
    pattern_module = MLIR.Module.create(ctx, pdl_pattern_str)
    if MLIR.Module.is_null(pattern_module), do: raise("fail to parse module")
    MLIR.Operation.verify!(pattern_module)
    pdl_pattern = CAPI.beaverPDLPatternGet(pattern_module)
    pdl_pattern
  end
end
