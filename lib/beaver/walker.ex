alias Beaver.Native

alias Beaver.MLIR

alias Beaver.MLIR.{Value, Operation, OpOperand, Region, Module, Attribute, Block, NamedAttribute, Identifier}

defmodule Beaver.Walker do
  alias Beaver.MLIR.CAPI
  @behaviour Access

  @moduledoc """
  Provides traversal capabilities for MLIR structures.

  This module implements traversal functionality for MLIR structures including:
  - `operations/1`
  - `results/1`
  - `successors/1`
  - `attributes/1`
  - `regions/1`

  It implements both the `Enumerable` protocol and the `Access` behavior to provide
  a familiar interface for working with MLIR structures.

  ### Depth-first, pre-order and post-order walking
  Allows traversing MLIR structures in depth-first order, visiting each node and its
  children before moving to siblings. Supports both pre-order (visit node before children) and post-order (visit children
  before node).

  ### Mutation Support
  Walker callbacks should only traverse and inspect IR. Use
  `Beaver.MLIR.PatternRewriter` inside rewrite patterns, or collect matches and
  mutate them afterward with `Beaver.MLIR.IRRewriter`. Keeping traversal and
  mutation separate prevents callbacks from retaining invalidated operations.

  ### Access Syntax
  - Access behavior to provide convenient attribute access:
  ```elixir
  op[:attr_name]
  op["attr_name"]
  ```
  - convenient access to get operands, results, regions
  ```
  operands(op)[0]
  ```
  """
  @type operation() :: Module.t() | Operation.t()
  @type container() :: operation() | Region.t() | Block.t() | NamedAttribute.t()
  @type element() :: operation() | Region.t() | Block.t() | Value.t() | NamedAttribute.t()
  @type element_module() :: Operation | Region | Block | Value | Attribute
  @type t :: %__MODULE__{
          container: container(),
          element_module: element_module(),
          get_num: (container() -> Native.I64.t() | integer()) | nil,
          get_element: (container(), integer() -> element()) | nil,
          get_first: (container() -> element()) | nil,
          get_next: (element() -> element()) | nil,
          is_null: (element() -> Native.Bool.t() | bool()) | nil
        }

  container_keys = [:container, :element_module]
  index_func_keys = [:get_num, :get_element]
  iter_func_keys = [:get_first, :get_next, :is_null]
  @enforce_keys container_keys
  defstruct container_keys ++ index_func_keys ++ iter_func_keys

  # operands, results, attributes of one operation
  defp verify_nesting!(Operation, Value), do: :ok
  defp verify_nesting!(Operation, {Identifier, Attribute}), do: :ok
  # regions of one operation
  defp verify_nesting!(Operation, Region), do: :ok
  # successor blocks of one operation
  defp verify_nesting!(Operation, Block), do: :ok
  # blocks in a region
  defp verify_nesting!(Region, Block), do: :ok
  # operations in a block
  defp verify_nesting!(Block, Operation), do: :ok
  # arguments of a block
  defp verify_nesting!(Block, Value), do: :ok
  defp verify_nesting!(Value, OpOperand), do: :ok

  defp verify_nesting!(container_module, element_module) do
    raise "not a legal 2-level structure could be walked in MLIR: #{inspect(container_module)}(#{inspect(element_module)})"
  end

  # Extract a container could be traversed by walker from an `Beaver.MLIR.Operation` or a `Beaver.MLIR.Module`.
  defp extract_container(%Module{} = module), do: Operation.from_module(module)

  defp extract_container(%{
         operands: %Beaver.Walker{container: container},
         attributes: %Beaver.Walker{container: container},
         results: %Beaver.Walker{container: container},
         successors: %Beaver.Walker{container: container},
         regions: %Beaver.Walker{container: container}
       }),
       do: container

  defp extract_container(container), do: container

  defp new(container, element_module, callbacks) do
    container = %container_module{} = extract_container(container)
    verify_nesting!(container_module, element_module)
    struct!(__MODULE__, [container: container, element_module: element_module] ++ callbacks)
  end

  @spec operands(operation()) :: Enumerable.t()
  @doc """
  Returns an enumerable of the operands of an `operation()`.
  """
  def operands(op),
    do: new(op, Value, get_num: &CAPI.mlirOperationGetNumOperands/1, get_element: &CAPI.mlirOperationGetOperand/2)

  @spec results(operation()) :: Enumerable.t()
  @doc """
  Returns an enumerable of the results of an `operation()`.
  """
  def results(op),
    do: new(op, Value, get_num: &CAPI.mlirOperationGetNumResults/1, get_element: &CAPI.mlirOperationGetResult/2)

  @spec regions(operation()) :: Enumerable.t()
  @doc """
  Returns an enumerable of the regions of an `operation()`.
  """
  def regions(op),
    do: new(op, Region, get_num: &CAPI.mlirOperationGetNumRegions/1, get_element: &CAPI.mlirOperationGetRegion/2)

  @spec successors(operation()) :: Enumerable.t()
  @doc """
  Returns an enumerable of the successor blocks of an `operation()`.
  """
  def successors(op),
    do: new(op, Block, get_num: &CAPI.mlirOperationGetNumSuccessors/1, get_element: &CAPI.mlirOperationGetSuccessor/2)

  @spec attributes(operation()) :: Enumerable.t()
  @doc """
  Returns inherent attributes followed by discardable attributes.

  Prefer `inherent_attributes/1` or `discardable_attributes/1` when the
  distinction is semantically relevant.
  """
  def attributes(op) do
    new(
      op,
      {Identifier, Attribute},
      get_num: fn operation ->
        native_integer(CAPI.beaverOperationGetNumInherentAttributes(operation)) +
          native_integer(CAPI.mlirOperationGetNumDiscardableAttributes(operation))
      end,
      get_element: fn o, i ->
        inherent_count = native_integer(CAPI.beaverOperationGetNumInherentAttributes(o))

        na =
          if i < inherent_count,
            do: CAPI.beaverOperationGetInherentAttribute(o, i),
            else: CAPI.mlirOperationGetDiscardableAttribute(o, i - inherent_count)

        named_attribute_tuple(na)
      end
    )
  end

  @spec inherent_attributes(operation()) :: Enumerable.t()
  @doc "Returns the attached inherent/property attributes of an operation."
  def inherent_attributes(op) do
    new(op, {Identifier, Attribute},
      get_num: &CAPI.beaverOperationGetNumInherentAttributes/1,
      get_element: fn operation, index ->
        operation |> CAPI.beaverOperationGetInherentAttribute(index) |> named_attribute_tuple()
      end
    )
  end

  @spec discardable_attributes(operation()) :: Enumerable.t()
  @doc "Returns the discardable metadata attributes of an operation."
  def discardable_attributes(op) do
    new(op, {Identifier, Attribute},
      get_num: &CAPI.mlirOperationGetNumDiscardableAttributes/1,
      get_element: fn operation, index ->
        operation |> CAPI.mlirOperationGetDiscardableAttribute(index) |> named_attribute_tuple()
      end
    )
  end

  defp named_attribute_tuple(named_attribute),
    do: {MLIR.NamedAttribute.name(named_attribute), MLIR.NamedAttribute.attribute(named_attribute)}

  defp native_integer(value) when is_integer(value), do: value
  defp native_integer(value), do: Native.to_term(value)

  @doc """
  Returns an enumerable of the arguments of an `Block.t()`
  """
  @spec arguments(Block.t()) :: Enumerable.t()
  def arguments(%Block{} = block),
    do: new(block, Value, get_num: &CAPI.mlirBlockGetNumArguments/1, get_element: &CAPI.mlirBlockGetArgument/2)

  @spec operations(Block.t()) :: Enumerable.t()
  @doc """
  Returns an enumerable of the operations of an `Block.t()`
  """
  def operations(%Block{} = block) do
    new(block, Operation,
      get_first: &CAPI.mlirBlockGetFirstOperation/1,
      get_next: &CAPI.mlirOperationGetNextInBlock/1,
      is_null: &MLIR.null?/1
    )
  end

  @spec blocks(Region.t()) :: Enumerable.t()
  @doc """
  Returns an enumerable of the blocks of an `Region.t()`
  """
  def blocks(%Region{} = region) do
    new(region, Block,
      get_first: &CAPI.mlirRegionGetFirstBlock/1,
      get_next: &CAPI.mlirBlockGetNextInRegion/1,
      is_null: &MLIR.null?/1
    )
  end

  @spec uses(Value.t()) :: Enumerable.t()
  @doc """
  Returns an enumerable of the uses of an `Value.t()`
  """
  def uses(%Value{} = value) do
    new(value, OpOperand,
      get_first: &CAPI.mlirValueGetFirstUse/1,
      get_next: &CAPI.mlirOpOperandGetNextUse/1,
      is_null: fn x -> CAPI.mlirOpOperandIsNull(x) |> Native.to_term() end
    )
  end

  @impl Access
  def fetch(%__MODULE__{element_module: {Identifier, Attribute}} = walker, key) do
    case Enum.find(walker, fn {name, _attribute} -> MLIR.to_string(name) == to_string(key) end) do
      {_, %Attribute{} = attr} -> {:ok, attr}
      nil -> :error
    end
  end

  def fetch(%__MODULE__{element_module: element} = walker, key)
      when is_integer(key) do
    case Enum.at(walker, key) do
      %^element{} = value -> {:ok, value}
      nil -> :error
    end
  end

  @impl Access
  def get_and_update(_data, _key, _function), do: raise("get_and_update not supported")

  @impl Access
  def pop(_data, _key), do: raise("pop not supported")

  @type mlir() :: container() | element()

  @doc """
  Traverse a container in MLIR; it can be an operation, region, or block.
  You might expect this function works like `Macro.traverse/4`.
  ### More on manipulating the IR
  Keep mutation outside the active traversal. Use a pattern defined by
  `Beaver.MLIR.RewritePattern`, or first
  collect the operations to change and then use `Beaver.MLIR.IRRewriter`.
  ### Some tips
  - If your matching is very complicated, using `with/1` in Elixir should cover it.
  - Use `Rewrite.apply_patterns/3` for MLIR's benefit-based greedy pattern application.
  - You can run traversals in a MLIR pass by calling them in `run/1` so that it joins the general MLIR pass manager's orchestration and will be run in parallel when possible.
  """
  @spec traverse(mlir(), any(), (mlir(), any() -> {mlir(), any()}), (mlir(), any() -> {mlir(), any()})) ::
          {mlir(), any()}
  def traverse(mlir, acc, pre, post) when is_function(pre, 2) and is_function(post, 2) do
    mlir = extract_container(mlir)
    do_traverse(mlir, acc, pre, post)
  end

  # traverse the nested with the walker
  defp do_traverse(%__MODULE__{} = walker, acc, pre, post),
    do: do_traverse(Enum.to_list(walker), acc, pre, post)

  defp do_traverse(list, acc, pre, post) when is_list(list),
    do: :lists.mapfoldl(fn x, acc -> do_traverse(x, acc, pre, post) end, acc, list)

  defp do_traverse(%Operation{} = op, acc, pre, post) do
    {op, acc} = pre.(op, acc)

    {_operands, acc} = operands(op) |> Enum.map(fn value -> {:operand, value} end) |> do_traverse(acc, pre, post)

    {_attributes, acc} = attributes(op) |> do_traverse(acc, pre, post)

    {_results, acc} = results(op) |> Enum.map(fn value -> {:result, value} end) |> do_traverse(acc, pre, post)

    {_regions, acc} = regions(op) |> do_traverse(acc, pre, post)

    {_successors, acc} =
      successors(op) |> Enum.map(fn successor -> {:successor, successor} end) |> do_traverse(acc, pre, post)

    post.(op, acc)
  end

  defp do_traverse(%Region{} = region, acc, pre, post) do
    {region, acc} = pre.(region, acc)
    {_blocks, acc} = blocks(region) |> do_traverse(acc, pre, post)

    post.(region, acc)
  end

  defp do_traverse(%Block{} = block, acc, pre, post) do
    {block, acc} = pre.(block, acc)

    {_arguments, acc} = arguments(block) |> Enum.map(fn value -> {:argument, value} end) |> do_traverse(acc, pre, post)

    {_operations, acc} = operations(block) |> do_traverse(acc, pre, post)

    post.(block, acc)
  end

  defp do_traverse({:successor, %Block{}} = successor, acc, pre, post) do
    {successor, acc} = pre.(successor, acc)
    post.(successor, acc)
  end

  defp do_traverse({value_kind, %Value{}} = value, acc, pre, post)
       when value_kind in [:result, :operand, :argument] do
    {{^value_kind, %Value{}} = value, acc} = pre.(value, acc)
    post.(value, acc)
  end

  defp do_traverse(%NamedAttribute{} = named_attribute, acc, pre, post) do
    name = %Identifier{} = named_attribute |> MLIR.NamedAttribute.name()
    attribute = %Attribute{} = named_attribute |> MLIR.NamedAttribute.attribute()

    {{name, attribute}, acc} = pre.({name, attribute}, acc)
    {{name, attribute}, acc} = post.({name, attribute}, acc)
    named_attribute = MLIR.NamedAttribute.get(name, attribute)
    {named_attribute, acc}
  end

  defp do_traverse({%Identifier{} = name, %Attribute{} = attribute}, acc, pre, post) do
    {{name, attribute}, acc} = pre.({name, attribute}, acc)
    post.({name, attribute}, acc)
  end

  @doc """
  Performs a depth-first, pre-order traversal of a MLIR structure.
  """
  @spec prewalk(mlir(), (mlir() -> mlir())) :: mlir()
  def prewalk(ast, fun) when is_function(fun, 1), do: elem(prewalk(ast, nil, fn x, nil -> {fun.(x), nil} end), 0)

  @doc """
  Performs a depth-first, pre-order traversal of a MLIR structure using an accumulator.
  """
  @spec prewalk(mlir(), any, (mlir(), any -> {mlir(), any})) :: {mlir(), any}
  def prewalk(ast, acc, fun) when is_function(fun, 2), do: traverse(ast, acc, fun, fn x, a -> {x, a} end)

  @doc """
  Performs a depth-first, post-order traversal of a MLIR structure.
  """
  @spec postwalk(mlir(), (mlir() -> mlir())) :: mlir()
  def postwalk(ast, fun) when is_function(fun, 1), do: elem(postwalk(ast, nil, fn x, nil -> {fun.(x), nil} end), 0)

  @doc """
  Performs a depth-first, post-order traversal of a MLIR structure using an accumulator.
  """
  @spec postwalk(mlir(), any, (mlir(), any -> {mlir(), any})) :: {mlir(), any}
  def postwalk(ast, acc, fun) when is_function(fun, 2), do: traverse(ast, acc, fn x, a -> {x, a} end, fun)

  defimpl Enumerable do
    def count(%{get_num: get_num, container: container}) when is_function(get_num, 1) do
      value = get_num.(container)
      {:ok, if(is_integer(value), do: value, else: Native.to_term(value))}
    end

    def count(_walker), do: {:error, __MODULE__}
    def member?(walker, element), do: {:ok, Enum.any?(walker, &MLIR.equal?(&1, element))}

    def slice(%{get_element: get, container: container} = walker) when is_function(get, 2) do
      {:ok, count} = count(walker)
      {:ok, count, fn start, length, step -> for i <- 0..(length - 1)//1, do: get.(container, start + i * step) end}
    end

    def slice(_walker), do: {:error, __MODULE__}

    def reduce(_walker, {:halt, acc}, _fun), do: {:halted, acc}
    def reduce(walker, {:suspend, acc}, fun), do: {:suspended, acc, &reduce(walker, &1, fun)}
    def reduce(walker, acc, fun), do: Enumerable.reduce(stream(walker), acc, fun)

    defp stream(%{get_element: get, container: container} = walker) when is_function(get, 2) do
      {:ok, count} = count(walker)
      Stream.map(0..(count - 1)//1, &expect_element!(get.(container, &1), walker.element_module))
    end

    defp stream(walker) do
      Stream.unfold(walker.get_first.(walker.container), fn current ->
        current = expect_element!(current, walker.element_module)
        unless walker.is_null.(current), do: {current, walker.get_next.(current)}
      end)
    end

    defp expect_element!(%module{} = element, module), do: element
    defp expect_element!({%a{}, %b{}} = element, {a, b}), do: element
    defp expect_element!(element, expected), do: raise("Expected #{inspect(expected)}, got #{inspect(element)}")
  end
end
