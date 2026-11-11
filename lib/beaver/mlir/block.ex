alias Beaver.MLIR.{CAPI, Location, Type}
alias Beaver.{Deferred, Native}

defmodule Beaver.MLIR.Block do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR

  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native

  defp do_add_args!(block, ctx, default_loc, {t, loc})
       when is_struct(t, Deferred) or is_struct(loc, Deferred) do
    loc = if loc, do: Deferred.resolve(loc, ctx), else: default_loc
    CAPI.mlirBlockAddArgument(block, Deferred.resolve(t, ctx), loc)
  end

  defp do_add_args!(block, ctx, _default_loc, {t = %Type{}, loc}) do
    t = Deferred.resolve(t, ctx)
    loc = Deferred.resolve(loc, ctx)
    CAPI.mlirBlockAddArgument(block, t, loc)
  end

  defp do_add_args!(block, ctx, _default_loc, {t = {:parametric, _, _, _f}, loc}) do
    t = Deferred.resolve(t, ctx)
    loc = Deferred.resolve(loc, ctx)
    CAPI.mlirBlockAddArgument(block, t, loc)
  end

  defp do_add_args!(block, ctx, _default_loc, {t, loc}) when is_binary(t) do
    t = CAPI.mlirTypeParseGet(ctx, MLIR.StringRef.create(t))
    loc = Deferred.resolve(loc, ctx)
    CAPI.mlirBlockAddArgument(block, t, loc)
  end

  defp do_add_args!(block, ctx, default_loc, t),
    do: do_add_args!(block, ctx, default_loc, {t, default_loc})

  @type arg_type :: Type.t() | String.t()
  @type arg :: {arg_type(), Location.t()} | arg_type()
  @spec add_args!(__MODULE__.t(), list(arg), Keyword.t()) :: [MLIR.Value.t()]
  @doc """
  Adds arguments to `block` and returns their `MLIR.Value` handles.
  """
  def add_args!(block, args, opts \\ []) when is_list(args) do
    if args == [] do
      []
    else
      ctx = infer_arg_context!(block, args, opts)

      default_loc =
        case opts[:location] do
          nil -> CAPI.mlirLocationUnknownGet(ctx)
          loc -> Location.resolve(loc, ctx)
        end

      for arg <- args, do: do_add_args!(block, ctx, default_loc, arg)
    end
  end

  @doc "Adds a single argument to `block` and returns its `MLIR.Value` handle."
  @spec add_arg!(t(), arg(), keyword()) :: MLIR.Value.t()
  def add_arg!(%__MODULE__{} = block, arg, opts \\ []) do
    [val] = add_args!(block, [arg], opts)
    val
  end

  @doc """
  Adds `{name, type}` named arguments to `block` and returns a `%{name => MLIR.Value.t()}` map.
  """
  @spec add_named_args!(t(), [{atom() | String.t(), arg()}], keyword()) :: %{
          optional(atom() | String.t()) => MLIR.Value.t()
        }
  def add_named_args!(%__MODULE__{} = block, named_specs, opts \\ []) when is_list(named_specs) do
    {names, specs} = Enum.unzip(named_specs)
    values = add_args!(block, specs, opts)
    Map.new(Enum.zip(names, values))
  end

  defp infer_arg_context!(block, args, opts) do
    ctx =
      Deferred.context(opts) ||
        (opts[:location] && match?(%Location{}, opts[:location]) && MLIR.context(opts[:location])) ||
        Enum.find_value(args, fn
          t = %Type{} -> MLIR.context(t)
          {t = %Type{}, _} -> MLIR.context(t)
          {_, l = %Location{}} -> MLIR.context(l)
          _ -> nil
        end) ||
        case parent_operation(block) do
          %MLIR.Operation{} = op -> MLIR.context(op)
          nil -> nil
        end

    if !ctx, do: raise("Requires MLIR Context to add arguments. Otherwise, use types or locations already created.")
    ctx
  end

  def get_arg!(%__MODULE__{} = block, index), do: CAPI.mlirBlockGetArgument(block, index)

  @doc "Returns all block arguments as a list of `MLIR.Value` handles."
  @spec arguments(t()) :: [MLIR.Value.t()]
  def arguments(%__MODULE__{} = block), do: Beaver.Walker.arguments(block) |> Enum.to_list()

  @doc "Returns an enumerable of operations inside `block`."
  @spec operations(t()) :: Enumerable.t()
  def operations(%__MODULE__{} = block), do: Beaver.Walker.operations(block)

  @doc "Returns the first operation in `block`, or `nil` if empty."
  @spec first_operation(t()) :: MLIR.Operation.t() | nil
  def first_operation(%__MODULE__{} = block), do: block |> CAPI.mlirBlockGetFirstOperation() |> nil_if_null()

  @doc "Returns the terminator operation of `block`, or `nil` if absent."
  @spec terminator(t()) :: MLIR.Operation.t() | nil
  def terminator(%__MODULE__{} = block), do: block |> CAPI.mlirBlockGetTerminator() |> nil_if_null()

  @doc "Returns the parent operation owning `block`, or `nil` if detached."
  @spec parent_operation(t()) :: MLIR.Operation.t() | nil
  def parent_operation(%__MODULE__{} = block), do: block |> CAPI.mlirBlockGetParentOperation() |> nil_if_null()

  @doc "Returns the parent region containing `block`, or `nil` if detached."
  @spec parent_region(t()) :: MLIR.Region.t() | nil
  def parent_region(%__MODULE__{} = block), do: block |> CAPI.mlirBlockGetParentRegion() |> nil_if_null()

  defp nil_if_null(entity), do: if(MLIR.null?(entity), do: nil, else: entity)

  def create(arg_loc_pairs \\ []) when is_list(arg_loc_pairs) do
    {args, locs} = Enum.unzip(arg_loc_pairs)
    create(args, locs)
  end

  def create(args, [_ | _] = opts) when is_list(args) and is_tuple(hd(opts)) and is_atom(elem(hd(opts), 0)) do
    block = create([], [])
    if args != [], do: add_args!(block, args, opts)
    block
  end

  def create(args, locs) when is_list(args) and is_list(locs) do
    if length(args) != length(locs),
      do: raise("Different length of block args and types. Make sure the block/1 macro in call within mlir/1 macro")

    len = length(args)
    args = args |> Native.array(Type)
    locs = locs |> Native.array(Location)

    CAPI.mlirBlockCreate(len, args, locs)
  end

  @doc "Returns `true` if two `MLIR.Block` handles refer to the same underlying block."
  @spec equal?(t(), t()) :: boolean()
  def equal?(%__MODULE__{} = a, %__MODULE__{} = b),
    do: CAPI.mlirBlockEqual(a, b) |> Native.to_term()

  defdelegate destroy(blk), to: CAPI, as: :mlirBlockDestroy
  defdelegate append(blk, op), to: CAPI, as: :mlirBlockAppendOwnedOperation
end
