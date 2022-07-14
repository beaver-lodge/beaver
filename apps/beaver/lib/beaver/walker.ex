alias Beaver.MLIR

alias Beaver.MLIR.CAPI.{
  MlirModule,
  MlirOperation,
  MlirRegion,
  MlirAttribute,
  MlirBlock,
  MlirValue,
  MlirNamedAttribute,
  MlirType,
  MlirOperand
}

defmodule Beaver.Walker do
  alias Beaver.MLIR.CAPI

  defmodule OpReplacement do
    @type t() :: %__MODULE__{
            operands: Beaver.Walker.t() | list(),
            attributes: Beaver.Walker.t() | list(),
            results: Beaver.Walker.t() | list(),
            successors: Beaver.Walker.t() | list(),
            regions: Beaver.Walker.t() | list()
          }
    defstruct operands: [], attributes: [], results: [], successors: [], regions: []
  end

  @moduledoc """
  Walker to traverse MLIR structures including operands, results, successors, attributes, regions.
  It implements the `Enumerable` protocol and the `Access` behavior.
  """

  # TODO: traverse MlirNamedAttribute?
  @type operation() ::
          MlirModule.t() | MlirOperation.t() | OpReplacement.t() | Beaver.DSL.Op.Prototype.t()
  @type container() ::
          operation()
          | MlirRegion.t()
          | MlirBlock.t()
          | MlirNamedAttribute.t()

  @type element() ::
          operation()
          | MlirRegion.t()
          | MlirBlock.t()
          | MlirValue.t()
          | MlirNamedAttribute.t()

  @type element_module() :: MlirOperation | MlirRegion | MlirBlock | MlirValue | MlirAttribute

  @type t :: %__MODULE__{
          container: container(),
          element_module: element_module(),
          get_num: (container() -> Exotic.Value.t() | integer()) | nil,
          get_element: (container(), integer() | Exotic.Value.t() -> element()) | nil,
          element_equal: (element(), element() -> Exotic.Value.t() | integer()) | nil,
          get_first: (container() -> element()) | nil,
          get_next: (element() -> element()) | nil,
          get_parent: (element() -> container()) | nil,
          parent_equal: (element(), element() -> Exotic.Value.t() | integer()) | nil,
          is_null: (element() -> Exotic.Value.t() | bool()) | nil,
          this: element() | non_neg_integer() | nil,
          num: non_neg_integer() | nil
        }

  container_keys = [:container, :element_module]
  index_func_keys = [:get_num, :get_element, :element_equal]
  iter_keys = [:this, :num]
  iter_func_keys = [:get_first, :get_next, :get_parent, :is_null, :parent_equal]
  @enforce_keys container_keys
  defstruct container_keys ++ index_func_keys ++ iter_keys ++ iter_func_keys

  # operands, results, attributes of one operation
  defp verify_nesting!(MlirOperation, MlirValue), do: :ok
  defp verify_nesting!(OpReplacement, MlirValue), do: :ok
  defp verify_nesting!(MlirOperation, MlirNamedAttribute), do: :ok
  defp verify_nesting!(OpReplacement, MlirNamedAttribute), do: :ok
  # regions of one operation
  defp verify_nesting!(MlirOperation, MlirRegion), do: :ok
  defp verify_nesting!(OpReplacement, MlirRegion), do: :ok
  # successor blocks of one operation
  defp verify_nesting!(MlirOperation, MlirBlock), do: :ok
  defp verify_nesting!(OpReplacement, MlirBlock), do: :ok
  # blocks in a region
  defp verify_nesting!(MlirRegion, MlirBlock), do: :ok
  # operations in a block
  defp verify_nesting!(MlirBlock, MlirOperation), do: :ok
  # arguments of a block
  defp verify_nesting!(MlirBlock, MlirValue), do: :ok
  defp verify_nesting!(MlirValue, MlirOperand), do: :ok

  defp verify_nesting!(container_module, element_module) do
    raise "not a legal 2-level structure could be walked in MLIR: #{inspect(container_module)}(#{inspect(element_module)})"
  end

  def new(
        container,
        element_module,
        get_num: get_num,
        get_element: get_element,
        element_equal: element_equal
      )
      when is_function(get_num, 1) and
             is_function(get_element, 2) and
             is_function(element_equal, 2) do
    container = %container_module{} = Beaver.container(container)
    verify_nesting!(container_module, element_module)

    %__MODULE__{
      container: container,
      element_module: element_module,
      get_num: get_num,
      get_element: get_element,
      element_equal: element_equal
    }
  end

  def new(
        container,
        element_module,
        get_first: get_first,
        get_next: get_next,
        get_parent: get_parent,
        is_null: is_null
      )
      when is_function(get_first, 1) and
             is_function(get_next, 1) and
             is_function(get_parent, 1) and
             is_function(is_null, 1) do
    container = %container_module{} = Beaver.container(container)
    verify_nesting!(container_module, element_module)

    %__MODULE__{
      container: container,
      element_module: element_module,
      get_first: get_first,
      get_next: get_next,
      get_parent: get_parent,
      is_null: is_null
    }
  end

  @spec operands(operation()) :: Enumerable.t()
  def operands(%OpReplacement{operands: operands}) do
    operands
  end

  def operands(op) do
    new(
      op,
      MlirValue,
      get_num: &CAPI.mlirOperationGetNumOperands/1,
      get_element: &CAPI.mlirOperationGetOperand/2,
      element_equal: &CAPI.mlirValueEqual/2
    )
  end

  @spec results(operation()) :: Enumerable.t()
  def results(%OpReplacement{results: results}) do
    results
  end

  def results(op) do
    new(
      op,
      MlirValue,
      get_num: &CAPI.mlirOperationGetNumResults/1,
      get_element: &CAPI.mlirOperationGetResult/2,
      element_equal: &CAPI.mlirValueEqual/2
    )
  end

  @spec regions(operation()) :: Enumerable.t()
  def regions(%OpReplacement{regions: regions}) do
    regions
  end

  def regions(op) do
    new(
      op,
      MlirRegion,
      get_num: &CAPI.mlirOperationGetNumRegions/1,
      get_element: &CAPI.mlirOperationGetRegion/2,
      element_equal: &CAPI.mlirRegionEqual/2
    )
  end

  @spec successors(operation()) :: Enumerable.t()
  def successors(%OpReplacement{successors: successors}) do
    successors
  end

  def successors(op) do
    new(
      op,
      MlirBlock,
      get_num: &CAPI.mlirOperationGetNumSuccessors/1,
      get_element: &CAPI.mlirOperationGetSuccessor/2,
      element_equal: &CAPI.mlirBlockEqual/2
    )
  end

  @spec attributes(operation()) :: Enumerable.t()
  def attributes(%OpReplacement{attributes: attributes}) do
    attributes
  end

  def attributes(op) do
    new(
      op,
      MlirNamedAttribute,
      get_num: &CAPI.mlirOperationGetNumAttributes/1,
      get_element: &CAPI.mlirOperationGetAttribute/2,
      element_equal: &CAPI.mlirAttributeEqual/2
    )
  end

  @spec arguments(MlirBlock.t()) :: Enumerable.t()
  def arguments(%MlirBlock{} = block) do
    new(
      block,
      MlirValue,
      get_num: &CAPI.mlirBlockGetNumArguments/1,
      get_element: &CAPI.mlirBlockGetArgument/2,
      element_equal: &CAPI.mlirValueEqual/2
    )
  end

  @spec operations(MlirBlock.t()) :: Enumerable.t()
  def operations(%MlirBlock{} = block) do
    new(
      block,
      MlirOperation,
      get_first: &CAPI.mlirBlockGetFirstOperation/1,
      get_next: &CAPI.mlirOperationGetNextInBlock/1,
      get_parent: &CAPI.mlirOperationGetBlock/1,
      is_null: &MLIR.Operation.is_null/1
    )
  end

  @spec blocks(MlirRegion.t()) :: Enumerable.t()
  def blocks(%MlirRegion{} = region) do
    new(
      region,
      MlirBlock,
      get_first: &CAPI.mlirRegionGetFirstBlock/1,
      get_next: &CAPI.mlirBlockGetNextInRegion/1,
      get_parent: &CAPI.mlirBlockGetParentRegion/1,
      is_null: &MLIR.Block.is_null/1
    )
  end

  @spec uses(MlirValue.t()) :: Enumerable.t()
  def uses(%MlirValue{} = value) do
    new(
      value,
      MlirOperand,
      get_first: &CAPI.beaverValueGetFirstOperand/1,
      get_next: &CAPI.beaverOperandGetNext/1,
      get_parent: &CAPI.beaverOperandGetValue/1,
      is_null: fn x -> CAPI.beaverOperandIsNull(x) |> Exotic.Value.extract() end
    )
  end

  @behaviour Access
  @impl true
  def fetch(%__MODULE{element_module: MlirValue} = walker, key) when is_integer(key) do
    with %MlirValue{} = value <- Enum.at(walker, key) do
      {:ok, value}
    else
      nil -> :error
    end
  end

  def fetch(%__MODULE{element_module: MlirNamedAttribute} = walker, key) when is_binary(key) do
    found =
      walker
      |> Enum.find(fn named_attribute ->
        with name <-
               named_attribute
               |> Exotic.Value.fetch(MLIR.CAPI.MlirNamedAttribute, :name)
               |> MLIR.CAPI.mlirIdentifierStr()
               |> MLIR.StringRef.extract() do
          name == key
        end
      end)

    with %MlirNamedAttribute{} <- found do
      {:ok, Exotic.Value.fetch(found, MLIR.CAPI.MlirNamedAttribute, :attribute)}
    else
      :error -> :error
    end
  end

  @impl true
  def get_and_update(_data, _key, _function) do
    raise "get_and_update not supported"
  end

  @impl true
  def pop(_data, _key) do
    raise "pop not supported"
  end

  @type mlir() :: container() | element()

  @doc """
  Traverse and transform a container in MLIR, it could be a operation, region, block.
  You might expect this function works like `Macro.traverse/4`.
  ### More on manipulating the IR
  During the traversal, there are generally two choices to manipulate the IR:
  - Use `Beaver.concrete/1` to extract a op/attribute to a elixir structure, and generate a new `OpReplacement` with `replace/2`.
  - Use a pattern defined by macro `Beaver.defpat/2` to have the PDL interpreter transform the IR for you.
  You can use both if it is proper to do so.
  It could be mind-boggling to think the IR is mutable but not an issue if your approach is very functional. Inappropriate mutation might cause crash or bugs if somewhere else is keeping a reference of the replace op.
  ### Some tips
  - If your matching is very complicated, using `with/1` in Elixir should cover it.
  - Use `defpat` if you want MLIR's greedy pattern application based on benefits instead of implementing something alike yourself.
  - You can run traversals in a MLIR pass by calling them in `run/1` so that it joins the general MLIR pass manager's orchestration and will be run in parallel when possible.
  """
  @spec traverse(
          container(),
          any(),
          (container() | element(), any() -> {mlir(), any()}),
          (container() | element(), any() -> {mlir(), any()})
        ) ::
          {mlir(), any()}
  def traverse(mlir, acc, pre, post) when is_function(pre, 2) and is_function(post, 2) do
    mlir = Beaver.container(mlir)
    do_traverse(mlir, acc, pre, post)
  end

  # traverse the nested with the walker
  defp do_traverse(%__MODULE__{} = walker, acc, pre, post) do
    do_traverse(Enum.to_list(walker), acc, pre, post)
  end

  defp do_traverse(list, acc, pre, post) when is_list(list) do
    :lists.mapfoldl(
      fn x, acc ->
        do_traverse(x, acc, pre, post)
      end,
      acc,
      list
    )
  end

  defp do_traverse(%MlirOperation{} = operation, acc, pre, post) do
    {operation, acc} = pre.(operation, acc)

    {operands, acc} =
      operands(operation)
      |> Enum.map(fn value -> {:operand, value} end)
      |> do_traverse(acc, pre, post)

    {attributes, acc} = attributes(operation) |> do_traverse(acc, pre, post)

    {results, acc} =
      results(operation)
      |> Enum.map(fn value -> {:result, value} end)
      |> do_traverse(acc, pre, post)

    {regions, acc} = regions(operation) |> do_traverse(acc, pre, post)

    {successors, acc} =
      successors(operation)
      |> Enum.map(fn successor -> {:successor, successor} end)
      |> do_traverse(acc, pre, post)

    # operands
    # mlirOperationSetOperand

    # attributes
    # mlirOperationSetAttributeByName
    # mlirOperationRemoveAttributeByName

    # results/regions/successor
    # replace with new op

    post.(operation, acc)
  end

  defp do_traverse(%MlirRegion{} = region, acc, pre, post) do
    {region, acc} = pre.(region, acc)
    {blocks, acc} = blocks(region) |> do_traverse(acc, pre, post)

    post.(region, acc)
  end

  defp do_traverse(%MlirBlock{} = block, acc, pre, post) do
    {block, acc} = pre.(block, acc)

    {arguments, acc} =
      arguments(block)
      |> Enum.map(fn value -> {:argument, value} end)
      |> do_traverse(acc, pre, post)

    {operations, acc} = operations(block) |> do_traverse(acc, pre, post)

    # Note: Erlang now owns the removed operation. call erase
    # mlirOperationRemoveFromParent
    # mlirBlockDestroy
    post.(block, acc)
  end

  defp do_traverse({:successor, %MlirBlock{}} = successor, acc, pre, post) do
    {successor, acc} = pre.(successor, acc)
    post.(successor, acc)
  end

  defp do_traverse({value_kind, %MlirValue{}} = value, acc, pre, post)
       when value_kind in [:result, :operand, :argument] do
    {{^value_kind, %MlirValue{}} = value, acc} = pre.(value, acc)
    post.(value, acc)
  end

  defp do_traverse(%MlirNamedAttribute{} = named_attribute, acc, pre, post) do
    name =
      %MLIR.CAPI.MlirIdentifier{} =
      named_attribute |> Exotic.Value.fetch(MLIR.CAPI.MlirNamedAttribute, :name)

    attribute =
      %MLIR.CAPI.MlirAttribute{} =
      named_attribute |> Exotic.Value.fetch(MLIR.CAPI.MlirNamedAttribute, :attribute)

    {{name, attribute}, acc} = pre.({name, attribute}, acc)
    {{name, attribute}, acc} = post.({name, attribute}, acc)
    named_attribute = MLIR.CAPI.mlirNamedAttributeGet(name, attribute)
    {named_attribute, acc}
  end

  @doc """
  Performs a depth-first, pre-order traversal of a MLIR structure.
  """
  @spec prewalk(t, (t -> t)) :: t
  def prewalk(ast, fun) when is_function(fun, 1) do
    elem(prewalk(ast, nil, fn x, nil -> {fun.(x), nil} end), 0)
  end

  @doc """
  Performs a depth-first, pre-order traversal of a MLIR structure using an accumulator.
  """
  @spec prewalk(t, any, (t, any -> {t, any})) :: {t, any}
  def prewalk(ast, acc, fun) when is_function(fun, 2) do
    traverse(ast, acc, fun, fn x, a -> {x, a} end)
  end

  @doc """
  Performs a depth-first, post-order traversal of a MLIR structure.
  """
  @spec postwalk(mlir(), (mlir() -> mlir())) :: mlir()
  def postwalk(ast, fun) when is_function(fun, 1) do
    elem(postwalk(ast, nil, fn x, nil -> {fun.(x), nil} end), 0)
  end

  @doc """
  Performs a depth-first, post-order traversal of a MLIR structure using an accumulator.
  """
  @spec postwalk(mlir(), any, (mlir(), any -> {mlir(), any})) :: {mlir(), any}
  def postwalk(ast, acc, fun) when is_function(fun, 2) do
    traverse(ast, acc, fn x, a -> {x, a} end, fun)
  end

  @spec replace(MlirOperation.t(), [MlirValue.t()] | MlirValue.t()) :: OpReplacement.t()
  @doc """
  Replace a operation with a value
  """
  def replace(%MlirOperation{} = op, %MlirValue{} = value) do
    with results <- results(op),
         1 <- Enum.count(results),
         %CAPI.MlirValue{} = result = results[0] do
      for %Beaver.MLIR.CAPI.MlirOperand{} = operand <- uses(result) do
        op = CAPI.beaverOperandGetOwner(operand)
        pos = CAPI.beaverOperandGetNumber(operand)
        CAPI.mlirOperationSetOperand(op, pos, value)
      end

      %OpReplacement{
        results: [value]
      }
    else
      _ ->
        raise "Operation being replace should have one and only result"
    end
  end
end

alias Beaver.Walker

defimpl Enumerable, for: Walker do
  @spec count(Walker.t()) :: {:ok, non_neg_integer()} | {:error, module()}
  def count(%Walker{container: container, get_num: get_num}) when is_function(get_num, 1) do
    {:ok, get_num.(container) |> Exotic.Value.extract()}
  end

  def count(%Walker{container: %container_module{}}) do
    {:error, container_module}
  end

  @spec member?(Walker.t(), Walker.element()) :: {:ok, boolean()} | {:error, module()}
  def member?(
        walker = %Walker{element_equal: element_equal, element_module: element_module},
        %element_module{} = element
      )
      when is_function(element_equal, 2) do
    is_member =
      Enum.any?(walker, fn member -> element_equal.(member, element) |> Exotic.Value.extract() end)

    {:ok, is_member}
  end

  @spec member?(Walker.t(), Walker.element()) :: {:ok, boolean()} | {:error, module()}
  def member?(
        %Walker{
          get_parent: get_parent,
          parent_equal: parent_equal,
          element_module: element_module,
          container: container
        },
        %element_module{} = element
      )
      when is_function(get_parent, 1) and is_function(parent_equal, 2) do
    is_member = parent_equal.(container, get_parent.(element)) |> Exotic.Value.extract()
    {:ok, is_member}
  end

  @spec slice(Walker.t()) ::
          {:ok, size :: non_neg_integer(), Enumerable.slicing_fun()} | {:error, module()}
  def slice(walker = %Walker{container: container, get_element: get_element})
      when is_function(get_element, 2) do
    with {:ok, count} <- count(walker) do
      {:ok, count,
       fn start, length ->
         pos_range = start..(start + length - 1)

         for pos <- pos_range do
           get_element.(container, pos)
         end
       end}
    else
      error -> error
    end
  end

  @spec reduce(Walker.t(), Enumerable.acc(), Enumerable.reducer()) :: Enumerable.result()

  # Do nothing special receiving :halt or :suspend
  def reduce(_walker, {:halt, acc}, _fun), do: {:halted, acc}

  def reduce(walker, {:suspend, acc}, fun),
    do: {:suspended, acc, &reduce(walker, &1, fun)}

  # Reduce by index
  def reduce(
        %Walker{get_element: get_element, this: nil, num: nil} = walker,
        {:cont, acc},
        fun
      )
      when is_function(get_element, 2) do
    with {:ok, count} <- count(walker) do
      reduce(%Walker{walker | this: 0, num: count}, {:cont, acc}, fun)
    else
      error -> error
    end
  end

  def reduce(
        %Walker{
          container: container,
          element_module: element_module,
          get_element: get_element,
          this: pos,
          num: num
        } = walker,
        {:cont, acc},
        fun
      )
      when is_function(get_element, 2) and
             is_integer(pos) and pos >= 0 do
    if pos < num do
      value = get_element.(container, pos) |> expect_element!(element_module)
      reduce(%Walker{walker | this: pos + 1}, fun.(value, acc), fun)
    else
      {:done, acc}
    end
  end

  # reduce by following the link
  def reduce(
        %Walker{
          container: container,
          element_module: element_module,
          get_first: get_first,
          get_next: get_next,
          is_null: is_null,
          this: nil
        } = walker,
        {:cont, acc},
        fun
      )
      when is_function(get_first, 1) and is_function(get_next, 1) and is_function(is_null, 1) do
    this = get_first.(container) |> expect_element!(element_module)
    reduce(%Walker{walker | this: this}, {:cont, acc}, fun)
  end

  def reduce(
        %Walker{
          get_next: get_next,
          is_null: is_null,
          element_module: element_module,
          this: %element_module{} = this
        } = walker,
        {:cont, acc},
        fun
      )
      when is_function(get_next, 1) and is_function(is_null, 1) do
    if is_null.(this) do
      {:done, acc}
    else
      next = get_next.(this) |> expect_element!(element_module)
      reduce(%Walker{walker | this: next}, fun.(this, acc), fun)
    end
  end

  # TODO: this could be a macro erased in :prod
  defp expect_element!(%module{} = element, element_module) do
    if element_module != module do
      raise "Expected element module #{element_module}, got #{module}"
    else
      element
    end
  end
end
