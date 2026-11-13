alias Beaver.Native
alias Beaver.MLIR
alias Beaver.MLIR.{Value, Operation, OpOperand, Region, Module, Attribute, Block, NamedAttribute, Identifier}

defmodule Beaver.Walker do
  alias Beaver.MLIR.CAPI
  @behaviour Access
  @moduledoc """
  Enumerable and Access views over MLIR operations, values, attributes, regions and blocks. For example,
  `operands(op)[0]` reads an operand and `attributes(op)[:name]` reads an attribute.

  `prewalk/2,3` and `postwalk/2,3` traverse depth-first, before or after children. Callbacks should inspect IR;
  collect changes and apply them afterward with `MLIR.IRRewriter`, or use `MLIR.PatternRewriter` within a rewrite
  pattern. Mutating during traversal can invalidate borrowed handles.
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
          get_parent: (element() -> container()) | nil,
          parent_equal: (element(), element() -> Native.Bool.t() | bool()) | nil,
          is_null: (element() -> Native.Bool.t() | bool()) | nil,
          this: element() | non_neg_integer() | nil,
          num: non_neg_integer() | nil
        }
  container_keys = [:container, :element_module]
  index_func_keys = [:get_num, :get_element]
  iter_keys = [:this, :num]
  iter_func_keys = [:get_first, :get_next, :get_parent, :is_null, :parent_equal]
  @enforce_keys container_keys
  defstruct container_keys ++ index_func_keys ++ iter_keys ++ iter_func_keys

  for {container, elements} <- [
        {Operation, [Value, {Identifier, Attribute}, Region, Block]},
        {Region, [Block]},
        {Block, [Operation, Value]},
        {Value, [OpOperand]}
      ],
      element <- elements do
    defp verify_nesting!(unquote(container), unquote(Macro.escape(element))), do: :ok
  end

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

  defp new(container, element_module, functions) do
    container = %container_module{} = extract_container(container)
    verify_nesting!(container_module, element_module)
    struct!(__MODULE__, [container: container, element_module: element_module] ++ functions)
  end

  for {name, kind, count, getter} <- [
        {:operands, Value, :mlirOperationGetNumOperands, :mlirOperationGetOperand},
        {:results, Value, :mlirOperationGetNumResults, :mlirOperationGetResult},
        {:regions, Region, :mlirOperationGetNumRegions, :mlirOperationGetRegion},
        {:successors, Block, :mlirOperationGetNumSuccessors, :mlirOperationGetSuccessor}
      ] do
    @doc "Returns an enumerable of the #{name} of an operation."
    @spec unquote(name)(operation()) :: Enumerable.t()
    def unquote(name)(op),
      do: new(op, unquote(kind), get_num: &(CAPI.unquote(count) / 1), get_element: &(CAPI.unquote(getter) / 2))
  end

  @spec attributes(operation()) :: Enumerable.t()
  @doc """
  Returns inherent attributes followed by discardable attributes.

  Prefer `inherent_attributes/1` or `discardable_attributes/1` when the distinction is semantically relevant.
  """
  def attributes(op) do
    new(op, {Identifier, Attribute},
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

  for {name, count, getter, doc} <- [
        {:inherent_attributes, :beaverOperationGetNumInherentAttributes, :beaverOperationGetInherentAttribute,
         "Returns the attached inherent/property attributes of an operation."},
        {:discardable_attributes, :mlirOperationGetNumDiscardableAttributes, :mlirOperationGetDiscardableAttribute,
         "Returns the discardable metadata attributes of an operation."}
      ] do
    @doc doc
    @spec unquote(name)(operation()) :: Enumerable.t()
    def unquote(name)(op) do
      new(op, {Identifier, Attribute},
        get_num: &(CAPI.unquote(count) / 1),
        get_element: fn op, index -> op |> CAPI.unquote(getter)(index) |> named_attribute_tuple() end
      )
    end
  end

  defp named_attribute_tuple(named_attribute),
    do: {MLIR.NamedAttribute.name(named_attribute), MLIR.NamedAttribute.attribute(named_attribute)}

  defp native_integer(value) when is_integer(value), do: value
  defp native_integer(value), do: Native.to_term(value)
  @doc "Returns an enumerable of the arguments of an `Block.t()`"
  @spec arguments(Block.t()) :: Enumerable.t()
  def arguments(%Block{} = block),
    do: new(block, Value, get_num: &CAPI.mlirBlockGetNumArguments/1, get_element: &CAPI.mlirBlockGetArgument/2)

  for {name, container, element, first, next, parent} <- [
        {:operations, Block, Operation, :mlirBlockGetFirstOperation, :mlirOperationGetNextInBlock,
         :mlirOperationGetBlock},
        {:blocks, Region, Block, :mlirRegionGetFirstBlock, :mlirBlockGetNextInRegion, :mlirBlockGetParentRegion},
        {:uses, Value, OpOperand, :mlirValueGetFirstUse, :mlirOpOperandGetNextUse, :mlirOpOperandGetValue}
      ] do
    @doc "Returns an enumerable of the #{name} of a `#{inspect(container)}.t()`."
    @spec unquote(name)(unquote(container).t()) :: Enumerable.t()
    def unquote(name)(%unquote(container){} = container) do
      new(container, unquote(element),
        get_first: &(CAPI.unquote(first) / 1),
        get_next: &(CAPI.unquote(next) / 1),
        get_parent: &(CAPI.unquote(parent) / 1),
        is_null: &element_null?/1
      )
    end
  end

  defp element_null?(%OpOperand{} = operand), do: CAPI.mlirOpOperandIsNull(operand) |> Native.to_term()
  defp element_null?(element), do: MLIR.null?(element)
  @impl Access
  def fetch(%__MODULE__{element_module: kind} = walker, key)
      when kind in [NamedAttribute, {Identifier, Attribute}] do
    key = to_string(key)

    Enum.find_value(walker, :error, fn entry ->
      {name, attr} = attribute_pair(entry)
      if MLIR.to_string(name) == key, do: {:ok, attr}
    end)
  end

  def fetch(%__MODULE__{element_module: element} = walker, key)
      when is_integer(key) do
    case Enum.at(walker, key) do
      %^element{} = value -> {:ok, value}
      nil -> :error
    end
  end

  defp attribute_pair(%NamedAttribute{} = entry), do: named_attribute_tuple(entry)
  defp attribute_pair({%Identifier{}, %Attribute{}} = entry), do: entry
  @impl Access
  def get_and_update(_data, _key, _function), do: raise("get_and_update not supported")
  @impl Access
  def pop(_data, _key), do: raise("pop not supported")
  @type mlir() :: container() | element()
  @doc """
  Traverses operations, regions and blocks like `Macro.traverse/4`. `pre` runs before children and `post` after
  them; keep mutation outside the traversal.
  """
  @spec traverse(mlir(), any(), (mlir(), any() -> {mlir(), any()}), (mlir(), any() -> {mlir(), any()})) ::
          {mlir(), any()}
  def traverse(mlir, acc, pre, post) when is_function(pre, 2) and is_function(post, 2) do
    mlir = extract_container(mlir)
    do_traverse(mlir, acc, pre, post)
  end

  # traverse the nested with the walker
  defp do_traverse(%__MODULE__{} = walker, acc, pre, post), do: do_traverse(Enum.to_list(walker), acc, pre, post)

  defp do_traverse(list, acc, pre, post) when is_list(list),
    do: :lists.mapfoldl(fn x, acc -> do_traverse(x, acc, pre, post) end, acc, list)

  for {kind, children} <- [
        {Operation, [operands: :operand, attributes: nil, results: :result, regions: nil, successors: :successor]},
        {Region, [blocks: nil]},
        {Block, [arguments: :argument, operations: nil]}
      ] do
    defp do_traverse(%unquote(kind){} = node, acc, pre, post) do
      {node, acc} = pre.(node, acc)

      acc =
        Enum.reduce(unquote(children), acc, fn {accessor, tag}, acc ->
          children = apply(__MODULE__, accessor, [node])
          children = if tag, do: Enum.map(children, &{tag, &1}), else: children
          {_children, acc} = do_traverse(children, acc, pre, post)
          acc
        end)

      post.(node, acc)
    end
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
    {{name, attribute}, acc} = do_traverse(named_attribute_tuple(named_attribute), acc, pre, post)
    {MLIR.NamedAttribute.get(name, attribute), acc}
  end

  defp do_traverse({%Identifier{} = name, %Attribute{} = attribute}, acc, pre, post) do
    {{name, attribute}, acc} = pre.({name, attribute}, acc)
    post.({name, attribute}, acc)
  end

  @doc "Performs a depth-first, pre-order traversal of a MLIR structure."
  @spec prewalk(mlir(), (mlir() -> mlir())) :: mlir()
  def prewalk(ast, fun) when is_function(fun, 1), do: elem(prewalk(ast, nil, fn x, nil -> {fun.(x), nil} end), 0)
  @doc "Performs a depth-first, pre-order traversal using an accumulator."
  @spec prewalk(mlir(), any, (mlir(), any -> {mlir(), any})) :: {mlir(), any}
  def prewalk(ast, acc, fun) when is_function(fun, 2), do: traverse(ast, acc, fun, fn x, a -> {x, a} end)
  @doc "Performs a depth-first, post-order traversal of a MLIR structure."
  @spec postwalk(mlir(), (mlir() -> mlir())) :: mlir()
  def postwalk(ast, fun) when is_function(fun, 1), do: elem(postwalk(ast, nil, fn x, nil -> {fun.(x), nil} end), 0)
  @doc "Performs a depth-first, post-order traversal using an accumulator."
  @spec postwalk(mlir(), any, (mlir(), any -> {mlir(), any})) :: {mlir(), any}
  def postwalk(ast, acc, fun) when is_function(fun, 2), do: traverse(ast, acc, fn x, a -> {x, a} end, fun)

  @type op_matcher() ::
          :all | String.t() | atom() | [String.Chars.t()] | MapSet.t() | (Operation.t() -> as_boolean(term()))
  @doc """
  Finds operations inside `root` (`MLIR.Module`, `MLIR.Operation`, `MLIR.Region`, or `MLIR.Block`)
  in pre-order without traversing operands, results, attributes, or successors.

  `matcher` may be `:all`, an operation name (`"arith.addi"`), a list or `MapSet` of operation names,
  or a 1-arity predicate `(Operation.t() -> boolean())`.

  Options:
    * `:exclude_root` - when `true`, skips `root` itself (default `false`; note that when `root`
      is an `MLIR.Module`, `builtin.module` is skipped when `exclude_root: true`).
  """
  @spec find_ops(Module.t() | Operation.t() | Region.t() | Block.t(), op_matcher(), keyword()) :: [Operation.t()]
  def find_ops(root, matcher \\ :all, opts \\ []) when is_list(opts) do
    predicate = build_op_predicate!(matcher)
    exclude_root = Keyword.get(opts, :exclude_root, false)
    container = extract_container(root)

    children =
      case container do
        %Operation{} = op when exclude_root -> Enum.to_list(regions(op))
        node -> [node]
      end

    collect_ops(children, [], predicate) |> Enum.reverse()
  end

  defp build_op_predicate!(:all), do: fn _op -> true end
  defp build_op_predicate!(fun) when is_function(fun, 1), do: fn op -> if(fun.(op), do: true, else: false) end
  defp build_op_predicate!(names) when is_binary(names) or is_atom(names), do: build_op_predicate!([names])

  defp build_op_predicate!(names) when is_list(names) or is_struct(names, MapSet) do
    names = MapSet.new(names, &to_string/1)
    fn op -> MapSet.member?(names, Operation.name(op)) end
  end

  defp build_op_predicate!(other),
    do: raise(ArgumentError, "invalid Walker.find_ops matcher: #{inspect(other)}")

  defp collect_ops(nodes, acc, predicate) do
    Enum.reduce(nodes, acc, fn
      %Operation{} = node, acc ->
        collect_ops(regions(node), if(predicate.(node), do: [node | acc], else: acc), predicate)

      %Region{} = node, acc ->
        collect_ops(blocks(node), acc, predicate)

      %Block{} = node, acc ->
        collect_ops(operations(node), acc, predicate)
    end)
  end

  defimpl Enumerable do
    @spec count(Beaver.Walker.t()) :: {:ok, non_neg_integer()} | {:error, module()}
    def count(%Beaver.Walker{container: container, get_num: get_num})
        when is_function(get_num, 1) do
      count = get_num.(container)
      {:ok, if(is_integer(count), do: count, else: Native.to_term(count))}
    end

    def count(%Beaver.Walker{}), do: {:error, __MODULE__}
    @spec member?(Beaver.Walker.t(), Beaver.Walker.element()) :: {:ok, boolean()} | {:error, module()}
    def member?(%Beaver.Walker{element_module: element_module} = walker, %element_module{} = element),
      do: {:ok, Enum.any?(walker, &MLIR.equal?(&1, element))}

    @spec slice(Beaver.Walker.t()) :: {:ok, size :: non_neg_integer(), Enumerable.slicing_fun()} | {:error, module()}
    def slice(walker = %Beaver.Walker{container: container, get_element: get_element})
        when is_function(get_element, 2) do
      case count(walker) do
        {:ok, count} -> {:ok, count, slicing_fun(container, get_element)}
        error -> error
      end
    end

    def slice(%Beaver.Walker{get_first: get_first, get_next: get_next})
        when is_function(get_first, 1) and is_function(get_next, 1), do: {:error, __MODULE__}

    defp slicing_fun(container, get_element),
      do: fn start, length, step -> for pos <- start..(start + length - 1)//step, do: get_element.(container, pos) end

    @spec reduce(Beaver.Walker.t(), Enumerable.acc(), Enumerable.reducer()) :: Enumerable.result()
    # Do nothing special receiving :halt or :suspend
    def reduce(_walker, {:halt, acc}, _fun), do: {:halted, acc}
    def reduce(walker, {:suspend, acc}, fun), do: {:suspended, acc, &reduce(walker, &1, fun)}
    def reduce(%Beaver.Walker{this: nil} = walker, {:cont, acc}, fun), do: reduce(initialize(walker), {:cont, acc}, fun)

    def reduce(walker, {:cont, acc}, fun) do
      case next(walker) do
        :done -> {:done, acc}
        {value, next} -> reduce(next, fun.(value, acc), fun)
      end
    end

    defp initialize(%{get_element: get_element} = walker) when is_function(get_element, 2) do
      {:ok, count} = count(walker)
      %{walker | this: 0, num: count}
    end

    defp initialize(walker),
      do: %{walker | this: expect_element!(walker.get_first.(walker.container), walker.element_module)}

    defp next(%{this: index, num: count} = walker) when is_integer(index) do
      if index < count do
        value = expect_element!(walker.get_element.(walker.container, index), walker.element_module)
        {value, %{walker | this: index + 1}}
      else
        :done
      end
    end

    defp next(walker) do
      if walker.is_null.(walker.this) do
        :done
      else
        next = expect_element!(walker.get_next.(walker.this), walker.element_module)
        {walker.this, %{walker | this: next}}
      end
    end

    defp expect_element!(%module{} = element, module), do: element
    defp expect_element!(%module{}, expected), do: raise("Expected element module #{expected}, got #{module}")
    defp expect_element!({%module_a{}, %module_b{}} = element, {module_a, module_b}), do: element

    defp expect_element!({%module_a{}, %module_b{}}, expected),
      do: raise("Expected element module #{inspect(expected)}, got #{inspect({module_a, module_b})}")
  end
end
