alias Beaver.MLIR.CAPI

defmodule Beaver.MLIR.RewritePatternSet do
  @moduledoc "This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}."
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Beaver.Native
  alias Beaver.MLIR

  defdelegate create(context), to: CAPI, as: :mlirRewritePatternSetCreate

  @doc """
  Create a `MLIR.RewritePatternSet` from the given PDL patterns. The `patterns` is a list of functions that accept
  two arguments: `ctx` and `block`, and create PDL patterns in the given `block` under the given `ctx`. The `opts`
  may contain: * `:ctx` - (required) the `MLIR.Context` to create the patterns in. * `:debug` - (optional) if set to
  `true`, dump each created pattern for debugging.
  """
  def with_pdl_patterns(patterns, opts \\ []) do
    pattern_module = build_pdl_module(patterns, opts)
    pdl = CAPI.mlirPDLPatternModuleFromModule(pattern_module)
    {CAPI.mlirRewritePatternSetFromPDLPatternModule(pdl), pattern_module}
  end

  @doc """
  Add a rewrite pattern into a `MLIR.RewritePatternSet`.

  The set takes ownership of a raw `MLIR.RewritePattern`. Native descriptors are materialized through the existing
  callback-backed rewrite bridge before ownership is transferred. Native programs execute entirely in the rewrite
  driver.
  """
  @spec add(t(), MLIR.RewritePattern.t()) :: t()
  def add(%__MODULE__{} = set, %MLIR.RewritePattern{} = pattern) do
    CAPI.mlirRewritePatternSetAdd(set, pattern)
    set
  end

  @spec add(t(), Beaver.Pattern.Native.Descriptor.t() | Beaver.Pattern.Native.Program.t(), keyword()) :: t()
  def add(%__MODULE__{} = set, %Beaver.Pattern.Native.Program{} = program, opts) do
    ctx = Keyword.fetch!(opts, :ctx)

    ctx_arg =
      case Keyword.get(opts, :type_converter) do
        %MLIR.TypeConverter{ref: converter_ref} -> {ctx.ref, converter_ref}
        nil -> ctx.ref
      end

    :ok =
      CAPI.beaver_raw_declarative_pattern_add(
        set.ref,
        ctx_arg,
        program.root,
        program.benefit,
        program.checks,
        program.edits
      )

    set
  end

  def add(%__MODULE__{} = set, %Beaver.Pattern.Native.Descriptor{} = descriptor, opts)
      when is_list(opts) do
    ctx = Keyword.get(opts, :ctx) || raise ArgumentError, "option :ctx is required"
    descriptor = Beaver.Pattern.Native.configure(descriptor, Keyword.delete(opts, :ctx))

    MLIR.RewritePattern.create(descriptor.root,
      ctx: ctx,
      benefit: descriptor.benefit,
      init_state: descriptor.init_state,
      construct: descriptor.construct,
      destruct: descriptor.destruct,
      prefilter: descriptor.prefilter,
      match_and_rewrite: descriptor.match_and_rewrite
    )
    |> then(&add(set, &1))
  end

  @spec add(t(), term(), module() | function() | Beaver.Pattern.Native.Descriptor.t()) :: t()
  def add(%__MODULE__{} = set, root_name, pattern), do: add(set, root_name, pattern, [])

  @spec add(t(), term(), module() | function() | Beaver.Pattern.Native.Descriptor.t(), keyword()) :: t()
  def add(%__MODULE__{} = set, root_name, %Beaver.Pattern.Native.Descriptor{} = descriptor, opts)
      when is_list(opts), do: add(set, descriptor, Keyword.put(opts, :root, root_name))

  def add(%__MODULE__{} = set, root_name, module, opts) when is_atom(module) and is_list(opts) do
    opts = if function_exported?(module, :construct, 1), do: put_in(opts, [:construct], &module.construct/1), else: opts

    opts = if function_exported?(module, :destruct, 1), do: put_in(opts, [:destruct], &module.destruct/1), else: opts
    opts = put_in(opts, [:match_and_rewrite], &module.match_and_rewrite/4)
    MLIR.RewritePattern.create(root_name, opts) |> then(&add(set, &1))
  end

  def add(%__MODULE__{} = set, root_name, match_and_rewrite, opts)
      when is_function(match_and_rewrite, 4) and is_list(opts) do
    benefit = opts[:benefit] || 1
    ctx = opts[:ctx] || raise "ctx is required in opts"

    MLIR.RewritePattern.create(root_name, ctx: ctx, benefit: benefit, match_and_rewrite: match_and_rewrite)
    |> then(&add(set, &1))
  end

  @doc """
  Adds a PDL pattern module (or PDL source string / builder callbacks) into an existing `MLIR.RewritePatternSet`.
  When frozen, MLIR compiles the PDL patterns into `PDLInterp` bytecode so matching and rewriting execute natively
  without BEAM callbacks.
  """
  @spec add_pdl(t(), MLIR.Module.t() | MLIR.Operation.t() | binary() | [function()], keyword()) :: t()
  def add_pdl(set, pdl, opts \\ [])

  def add_pdl(%__MODULE__{} = set, %kind{} = module, _opts) when kind in [MLIR.Module, MLIR.Operation] do
    :ok = CAPI.beaver_raw_pdl_pattern_add(set.ref, MLIR.Operation.from_module(module).ref)
    set
  end

  def add_pdl(%__MODULE__{} = set, pdl_source, opts) when is_binary(pdl_source) do
    ctx = Keyword.get(opts, :ctx) || raise ArgumentError, "option :ctx is required when adding PDL source string"

    Beaver.Native.with_resource(MLIR.Module.create!(pdl_source, ctx: ctx), &MLIR.Module.destroy/1, fn pdl_module ->
      add_pdl(set, pdl_module, opts)
    end)
  end

  def add_pdl(%__MODULE__{} = set, patterns, opts) when is_list(patterns) do
    Beaver.Native.with_resource(build_pdl_module(patterns, opts), &MLIR.Module.destroy/1, &add_pdl(set, &1, opts))
  end

  defp build_pdl_module(patterns, opts) do
    ctx = opts[:ctx] || raise ArgumentError, "option :ctx is required when building PDL patterns"
    module = MLIR.Location.from_env(__ENV__, ctx: ctx) |> MLIR.Module.empty()

    try do
      block = MLIR.Module.body(module)

      for build <- patterns do
        op = build.(ctx, block)
        if opts[:debug], do: MLIR.dump!(op)
      end

      MLIR.verify!(module)
    catch
      kind, reason ->
        MLIR.Module.destroy(module)
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  @doc "Creates a dialect conversion pattern and transfers it into this set."
  def add_conversion(set, root_name, converter, match_and_rewrite, opts \\ []),
    do: MLIR.ConversionPattern.add(set, root_name, converter, match_and_rewrite, opts)

  defdelegate destroy(set), to: CAPI, as: :mlirRewritePatternSetDestroy

  @doc "Destroys a pattern set on the callback executor."
  def threaded_destroy(ctx, %kind{ref: ref}) when kind in [__MODULE__, MLIR.FrozenRewritePatternSet] do
    name = if kind == __MODULE__, do: :RewritePatternSet, else: :FrozenRewritePatternSet
    Beaver.Execution.run(ctx, :destroy, {name, ref})
  end

  @doc """
  Freeze the given `MLIR.RewritePatternSet` to a `MLIR.FrozenRewritePatternSet`. Note that the ownership of the
  input set is transferred into the frozen set after this call.
  """
  defdelegate freeze(set), to: CAPI, as: :mlirFreezeRewritePattern
end
