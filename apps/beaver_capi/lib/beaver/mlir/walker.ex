alias Beaver.MLIR.CAPI.{
  MlirModule,
  MlirOperation,
  MlirRegion,
  MlirAttribute,
  MlirBlock,
  MlirValue
}

defmodule Beaver.MLIR.Walker do
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Walkers traverses MLIR structures including operands, results, successors, attributes, regions
  """

  @type container() :: MlirOperation.t() | MlirRegion.t() | MlirBlock.t()

  @type container_module() :: MlirOperation | MlirRegion | MlirBlock

  @type element() ::
          MlirOperation.t() | MlirRegion.t() | MlirBlock.t() | MlirValue.t() | MlirAttribute.t()

  @type element_module() :: MlirOperation | MlirRegion | MlirBlock | MlirValue | MlirAttribute

  @type t :: %__MODULE__{
          container: container(),
          container_module: element_module(),
          element_module: element_module(),
          get_num: (container() -> Exotic.Value.t() | integer()) | nil,
          get_element: (container(), integer() | Exotic.Value.t() -> element()) | nil,
          element_equal: (element(), element() -> Exotic.Value.t() | integer()) | nil,
          get_first: (container() -> element()) | nil,
          get_next: (element() -> element()) | nil,
          get_parent: (element() -> container()) | nil
        }

  container_keys = [:container, :container_module, :element_module]
  index_func_keys = [:get_num, :get_element, :element_equal]
  iter_func_keys = [:get_first, :get_next, :get_parent]
  @enforce_keys [:container, :container_module, :element_module]
  defstruct container_keys ++ index_func_keys ++ iter_func_keys

  # operands, results, attributes of one operation
  defp verify_nesting!(MlirOperation, MlirValue), do: :ok
  defp verify_nesting!(MlirOperation, MlirAttribute), do: :ok
  # regions of one operation
  defp verify_nesting!(MlirOperation, MlirRegion), do: :ok
  # successor blocks of a operation
  defp verify_nesting!(MlirOperation, MlirBlock), do: :ok
  # blocks in a regions
  defp verify_nesting!(MlirRegion, MlirBlock), do: :ok
  # operations in a block
  defp verify_nesting!(MlirBlock, MlirOperation), do: :ok
  # arguments of a block
  defp verify_nesting!(MlirBlock, MlirValue), do: :ok

  defp verify_nesting!(container_module, element_module) do
    raise "not a legal structure could be walked in MLIR: #{inspect(container_module)}(#{inspect(element_module)})"
  end

  defp to_container(module = %MlirModule{}) do
    CAPI.mlirModuleGetOperation(module)
  end

  defp to_container(container) do
    container
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
    container = %container_module{} = to_container(container)
    verify_nesting!(container_module, element_module)

    %__MODULE__{
      container: container,
      container_module: container_module,
      element_module: element_module,
      get_num: get_num,
      get_element: get_element,
      element_equal: get_element
    }
  end

  def new(
        container,
        element_module,
        get_first: get_first,
        get_next: get_next,
        get_parent: get_parent
      )
      when is_function(get_first, 1) and
             is_function(get_next, 1) and
             is_function(get_parent, 1) do
    container = %container_module{} = to_container(container)
    verify_nesting!(container_module, element_module)

    %__MODULE__{
      container: container,
      container_module: container_module,
      element_module: element_module,
      get_first: get_first,
      get_next: get_next,
      get_parent: get_parent
    }
  end

  @spec operands(MlirOperation.t()) :: Enumerable.result()
  def operands(op) do
    new(
      op,
      MlirValue,
      get_num: &CAPI.mlirOperationGetNumOperands/1,
      get_element: &CAPI.mlirOperationGetOperand/2,
      element_equal: &CAPI.mlirValueEqual/2
    )
  end

  @spec results(MlirOperation.t()) :: Enumerable.result()
  def results(op) do
    new(
      op,
      MlirValue,
      get_num: &CAPI.mlirOperationGetNumResults/1,
      get_element: &CAPI.mlirOperationGetResult/2,
      element_equal: &CAPI.mlirValueEqual/2
    )
  end

  @spec regions(MlirOperation.t()) :: Enumerable.result()
  def regions(op) do
    new(
      op,
      MlirValue,
      get_num: &CAPI.mlirOperationGetNumRegions/1,
      get_element: &CAPI.mlirOperationGetRegion/2,
      element_equal: &CAPI.mlirRegionEqual/2
    )
  end

  @spec successors(MlirOperation.t()) :: Enumerable.result()
  def successors(op) do
    new(
      op,
      MlirBlock,
      get_num: &CAPI.mlirOperationGetNumSuccessors/1,
      get_element: &CAPI.mlirOperationGetSuccessor/2,
      element_equal: &CAPI.mlirBlockEqual/2
    )
  end

  @spec attributes(MlirOperation.t()) :: Enumerable.result()
  def attributes(op) do
    new(
      op,
      MlirAttribute,
      get_num: &CAPI.mlirOperationGetNumAttributes/1,
      get_element: &CAPI.mlirOperationGetAttribute/2,
      element_equal: &CAPI.mlirAttributeEqual/2
    )
  end

  def arguments(block) do
    new(
      block,
      MlirValue,
      get_num: &CAPI.mlirBlockGetNumArguments/1,
      get_element: &CAPI.mlirBlockGetArgument/2,
      element_equal: &CAPI.mlirValueEqual/2
    )
  end

  def operations(block) do
  end

  def blocks(region) do
    new(
      region,
      MlirBlock,
      get_first: &CAPI.mlirRegionGetFirstBlock/1,
      get_next: &CAPI.mlirBlockGetNextInRegion/1,
      get_parent: &CAPI.mlirBlockGetParentRegion/1
    )
  end
end

alias Beaver.MLIR.Walker

defimpl Enumerable, for: Walker do
  def count(%Walker{container: container, get_num: get_num}) when is_function(get_num, 1) do
    get_num.(container) |> Exotic.Value.extract()
  end

  @spec member?(Walker.t(), Walker.element()) :: {:ok, boolean()} | {:error, module()}
  def member?(
        walker = %Walker{element_equal: element_equal, element_module: element_module},
        %element_module{} = element
      ) do
    is_member =
      Enum.any?(walker, fn member -> element_equal.(member, element) |> Exotic.Value.extract() end)

    {:ok, is_member}
  end

  @spec slice(Walker.t()) ::
          {:ok, size :: non_neg_integer(), Enumerable.slicing_fun()} | {:error, module()}
  def slice(walker = %Walker{container: container, get_element: get_element})
      when is_function(get_element, 2) do
    {:ok, count(walker),
     fn start, length ->
       pos_range = start..(start + length - 1)

       for pos <- pos_range do
         get_element.(container, pos)
       end
     end}
  end

  @spec reduce(Walker.t(), Enumerable.acc(), Enumerable.reducer()) :: Enumerable.result()

  # Do nothing special if :halt
  def reduce(_walker, {:halt, acc}, _fun), do: {:halted, acc}
  # Do nothing special if :suspend
  def reduce(walker, {:suspend, acc}, fun),
    do: {:suspended, acc, &reduce(walker, &1, fun)}

  # Reduce all in one :cont
  def reduce(%Walker{container: container, get_element: get_element} = walker, {:cont, acc}, fun)
      when is_function(get_element, 2) do
    pos_range = 0..(count(walker) - 1)//1

    Enum.reduce(pos_range, acc, fn pos, acc ->
      value = get_element.(container, pos)
      fun.(value, acc)
    end)
  end
end
