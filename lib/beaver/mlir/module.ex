alias Beaver.MLIR.Operation
alias Beaver.Deferred

defmodule Beaver.MLIR.Module do
  @moduledoc "This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}."
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  @behaviour Access

  @doc "Create a MLIR module by parsing string."
  def create(str, opts \\ []) when is_binary(str) do
    Deferred.from_opts(
      opts,
      fn ctx ->
        {module, diagnostics} = Beaver.Execution.run(ctx, :parse, {MLIR.StringRef.create(str).ref})

        if MLIR.null?(module), do: {:error, diagnostics}, else: {:ok, module}
      end
    )
  end

  def create!(str, opts \\ []) when is_binary(str) do
    res = Deferred.from_opts(opts, fn ctx -> Deferred.resolve(create(str, opts), ctx) end)

    case res do
      %Deferred{} -> raise ArgumentError, "calling a bang function to parse module must be eager"
      {:error, diagnostics} -> raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "fail to parse module")
      {:ok, module} -> MLIR.verify!(module)
    end
  end

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  defdelegate destroy(module), to: CAPI, as: :mlirModuleDestroy

  @doc "Gets the body block of `module` (or a single-region container `Operation`)."
  def body(%__MODULE__{} = module), do: CAPI.mlirModuleGetBody(module)

  def body(%Operation{} = op) do
    mod = CAPI.mlirModuleFromOperation(op)

    if MLIR.null?(mod) do
      CAPI.mlirOperationGetRegion(op, 0) |> MLIR.Region.first_block()
    else
      CAPI.mlirModuleGetBody(mod)
    end
  end

  defdelegate from_operation(op), to: CAPI, as: :mlirModuleFromOperation

  @doc "Creates an empty `MLIR.Module` from a `Location` or keyword options (`[ctx: ctx, location: loc]`)."
  def empty(opts \\ [])

  def empty(%MLIR.Location{} = location), do: CAPI.mlirModuleCreateEmpty(location)

  def empty(opts) when is_list(opts) do
    Deferred.from_opts(opts, fn ctx ->
      loc =
        case opts[:location] do
          nil -> MLIR.Location.unknown(ctx: ctx)
          loc -> MLIR.Location.resolve(loc, ctx)
        end

      CAPI.mlirModuleCreateEmpty(loc)
    end)
  end

  @doc "Parses, creates, or wraps an `MLIR.Module`, invokes `fun.(module)`, and always destroys the module afterward."
  @spec with_module(MLIR.Context.t() | binary() | t(), keyword() | (t() -> result)) :: result when result: var
  def with_module(%MLIR.Context{} = ctx, fun) when is_function(fun, 1),
    do: Beaver.Native.with_resource(empty(ctx: ctx), &destroy/1, fun)

  def with_module(source, fun) when is_function(fun, 1), do: with_module(source, [], fun)

  @spec with_module(MLIR.Context.t() | binary() | t(), binary() | keyword(), (t() -> result)) :: result when result: var
  def with_module(%MLIR.Context{} = ctx, str, fun) when is_binary(str) and is_function(fun, 1),
    do: Beaver.Native.with_resource(create!(str, ctx: ctx), &destroy/1, fun)

  def with_module(str, opts, fun) when is_binary(str) and is_list(opts) and is_function(fun, 1),
    do: Beaver.Native.with_resource(create!(str, opts), &destroy/1, fun)

  def with_module(%__MODULE__{} = module, _opts, fun) when is_function(fun, 1),
    do: Beaver.Native.with_resource(module, &destroy/1, fun)

  @impl Access
  def fetch(module, attribute), do: Operation.fetch(Operation.from_module(module), attribute)

  @impl Access
  def get_and_update(module, attribute, function) do
    {attr, _} = Operation.get_and_update(Operation.from_module(module), attribute, function)

    {attr, module}
  end

  @impl Access
  def pop(module, attribute) do
    {attr, _} = Operation.pop(Operation.from_module(module), attribute)
    {attr, module}
  end
end
