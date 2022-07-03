defmodule Beaver.DSL.Pattern do
  use Beaver

  # generate PDL ops for types and attributes
  def gen_pdl(%MLIR.CAPI.MlirType{} = type) do
    mlir do
      Beaver.MLIR.Dialect.PDL.type(type: type) >>> ~t{!pdl.type}
    end
  end

  def gen_pdl(%Beaver.MLIR.CAPI.MlirAttribute{} = attribute) do
    mlir do
      Beaver.MLIR.Dialect.PDL.attribute(value: attribute) >>>
        ~t{!pdl.attribute}
    end
  end

  def gen_pdl(%MLIR.CAPI.MlirValue{} = value) do
    value
  end

  @doc """
  The difference between a pdl.operation creation in a match body and a rewrite body:
  - in a match body, pdl.attribute/pdl.operand/pdl.result will be generated for unbound variables
  - in a rewrite body, all variables are considered bound before creation pdl ops
  """
  def create_operation(
        op_name,
        %Beaver.DSL.Op.Prototype{operands: operands, attributes: attributes, results: results},
        %Beaver.MLIR.CAPI.MlirAttribute{} = attribute_names
      ) do
    mlir do
      results = results |> Enum.map(&gen_pdl/1)
      attributes = attributes |> Enum.map(&gen_pdl/1)

      Beaver.MLIR.Dialect.PDL.operation(
        operands ++
          attributes ++
          results ++
          [
            name: Beaver.MLIR.Attribute.string(op_name),
            attributeNames: attribute_names,
            operand_segment_sizes:
              Beaver.MLIR.ODS.operand_segment_sizes([
                length(operands),
                length(attributes),
                length(results)
              ])
          ]
      ) >>> ~t{!pdl.operation}
    end
  end

  def gen_attribute_names(attributes_keys) do
    for key <- attributes_keys do
      key
      |> Atom.to_string()
      |> Beaver.MLIR.Attribute.string()
    end
    |> Beaver.MLIR.Attribute.array()
  end

  defp do_transform_match({:^, _, [var]}) do
    {:bound, var}
  end

  defp do_transform_match(
         {:%, _,
          [
            struct_name,
            {:%{}, _, map_args}
          ]} = _ast
       ) do
    operands = map_args |> Keyword.get(:operands, [])
    results = map_args |> Keyword.get(:results, [])
    attributes = map_args |> Keyword.get(:attributes, [])

    # Transform unbound variables of operands. If is bound, it means it is another operation's result (get bound in a previous expression)
    filtered_operands =
      Enum.map(operands, fn
        {:bound, bound} -> bound
        other -> other
      end)

    # extract unbound variables from attributes keyword
    filtered_attributes =
      Enum.map(attributes, fn
        {_k, {:bound, bound}} -> bound
        {_k, other} -> other
      end)

    map_args = Keyword.put(map_args, :operands, filtered_operands)
    map_args = Keyword.put(map_args, :attributes, filtered_attributes)

    # generate bounds between operand variables and pdl.operand
    operands =
      for operand <- operands do
        case operand do
          {:bound, _bound} ->
            quote do
              []
            end

          _ ->
            quote do
              unquote(operand) = Beaver.MLIR.Dialect.PDL.operand() >>> ~t{!pdl.value}
            end
        end
      end

    # generate bounds between attribute variables and pdl.attribute
    attributes_match =
      for {_name, attribute} <- attributes do
        case attribute do
          {:bound, bound} ->
            quote do
              unquote(bound) = Beaver.DSL.Pattern.gen_pdl(unquote(bound))
            end

          _ ->
            quote do
              unquote(attribute) = Beaver.MLIR.Dialect.PDL.attribute() >>> ~t{!pdl.attribute}
            end
        end
      end

    attributes_keys = Keyword.keys(attributes)

    # generate bounds between result variables and pdl.type
    results_match =
      for result <- results do
        quote do
          unquote(result) = Beaver.MLIR.Dialect.PDL.type() >>> ~t{!pdl.type}
        end
      end

    # generate bounds between result variables and pdl.value produced by pdl.result
    # preceding expression should use pin operator (^var) to access these version of binding otherwise it will get bound to a pdl.value produced by pdl.operand
    results_rebind =
      for {result, i} <- Enum.with_index(results) do
        quote do
          unquote(result) =
            Beaver.MLIR.Dialect.PDL.result(
              beaver_gen_root,
              index: Beaver.MLIR.Attribute.integer(Beaver.MLIR.Type.i32(), unquote(i))
            ) >>> ~t{!pdl.value}
        end
      end

    # injecting all variables and dispatch the op creation
    ast =
      quote do
        unquote_splicing(operands)
        unquote_splicing(attributes_match)
        unquote_splicing(results_match)

        attribute_names = Beaver.DSL.Pattern.gen_attribute_names(unquote(attributes_keys))

        beaver_gen_root =
          Beaver.DSL.Op.Prototype.dispatch(
            unquote(struct_name),
            unquote(map_args),
            attribute_names,
            &Beaver.DSL.Pattern.create_operation/3
          )

        unquote_splicing(results_rebind)
        beaver_gen_root
      end

    ast
  end

  defp do_transform_match(ast), do: ast

  def transform_match(ast) do
    Macro.postwalk(ast, &do_transform_match/1)
  end

  defp do_transform_rewrite(
         {:%, _,
          [
            struct_name,
            {:%{}, _, map_args}
          ]} = _ast
       ) do
    attributes = map_args |> Keyword.get(:attributes, [])

    filtered_attributes =
      Enum.map(attributes, fn
        {_k, other} -> other
      end)

    map_args = Keyword.put(map_args, :attributes, filtered_attributes)
    attributes_keys = Keyword.keys(attributes)

    ast =
      quote do
        attribute_names = Beaver.DSL.Pattern.gen_attribute_names(unquote(attributes_keys))

        Beaver.DSL.Op.Prototype.dispatch(
          unquote(struct_name),
          unquote(map_args),
          attribute_names,
          &Beaver.DSL.Pattern.create_operation/3
        )
      end

    ast
  end

  defp do_transform_rewrite(ast), do: ast

  @doc """
  transform a do block to PDL rewrite operation.
  Every Prototype form within the block should be transformed to create a PDL operation.
  The last expression will be replaced by the root op in the match by default.
  TODO: wrap this function with a macro rewrite/1, so that it could be use in a independent function
  """
  def transform_rewrite(ast) do
    rewrite_block_ast = Macro.postwalk(ast, &do_transform_rewrite/1)

    quote do
      Beaver.MLIR.Dialect.PDL.rewrite [
        beaver_gen_root,
        defer_if_terminator: false,
        operand_segment_sizes: Beaver.MLIR.ODS.operand_segment_sizes([1, 0])
      ] do
        region do
          block some() do
            repl = unquote(rewrite_block_ast)

            Beaver.MLIR.Dialect.PDL.replace([
              beaver_gen_root,
              repl,
              operand_segment_sizes: Beaver.MLIR.ODS.operand_segment_sizes([1, 1, 0])
            ]) >>> []
          end
        end
      end
    end
  end

  def result(%Beaver.MLIR.CAPI.MlirValue{} = v, i) when is_integer(i) do
    mlir do
      PDL.result(v, index: Beaver.MLIR.Attribute.integer(Beaver.MLIR.Type.i32(), i)) >>>
        ~t{!pdl.value}
    end
  end
end
