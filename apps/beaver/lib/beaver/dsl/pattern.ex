defmodule Beaver.DSL.Pattern do
  use Beaver

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

    filtered_operands =
      Enum.map(operands, fn
        {:bound, bound} -> bound
        other -> other
      end)

    filtered_attributes =
      Enum.map(attributes, fn
        {_k, {:bound, bound}} -> bound
        {_k, other} -> other
      end)

    map_args = Keyword.put(map_args, :operands, filtered_operands)
    map_args = Keyword.put(map_args, :attributes, filtered_attributes)

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

    attributes_match =
      for {_name, attribute} <- attributes do
        case attribute do
          {:bound, bound} ->
            quote do
              # TODO: match if it is a pdl attr or real attr
              unquote(bound) =
                case unquote(bound) do
                  %Beaver.MLIR.CAPI.MlirValue{} ->
                    unquote(bound)

                  %Beaver.MLIR.CAPI.MlirAttribute{} ->
                    Beaver.MLIR.Dialect.PDL.attribute(value: unquote(bound)) >>>
                      ~t{!pdl.attribute}
                end
            end

          _ ->
            quote do
              unquote(attribute) = Beaver.MLIR.Dialect.PDL.attribute() >>> ~t{!pdl.attribute}
            end
        end
      end

    attributes_keys = Keyword.keys(attributes)

    results_match =
      for result <- results do
        quote do
          unquote(result) = Beaver.MLIR.Dialect.PDL.type() >>> ~t{!pdl.type}
        end
      end

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

    ast =
      quote do
        unquote_splicing(operands)
        unquote_splicing(attributes_match)
        unquote_splicing(results_match)

        attribute_names =
          for key <- unquote(attributes_keys) do
            key
            |> Atom.to_string()
            |> Beaver.MLIR.Attribute.string()
          end
          |> Beaver.MLIR.Attribute.array()

        beaver_gen_root =
          Beaver.DSL.Op.Prototype.dispatch(
            unquote(struct_name),
            unquote(map_args),
            attribute_names,
            &Beaver.DSL.Pattern.create_root/3
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
    ast =
      quote do
        Beaver.DSL.Op.Prototype.dispatch(
          unquote(struct_name),
          unquote(map_args),
          beaver_gen_root,
          &Beaver.DSL.Pattern.create_replace/3
        )
      end

    ast
  end

  defp do_transform_rewrite(ast), do: ast

  def create_root(
        op_name,
        %Beaver.DSL.Op.Prototype{operands: operands, attributes: attributes, results: results},
        %Beaver.MLIR.CAPI.MlirAttribute{} = attribute_names
      ) do
    mlir do
      arguments = operands ++ attributes ++ results

      Beaver.MLIR.Dialect.PDL.operation(
        arguments ++
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

  def create_replace(
        op_name,
        %Beaver.DSL.Op.Prototype{operands: operands, attributes: attributes} = prototype,
        beaver_gen_root
      ) do
    mlir do
      repl =
        Beaver.MLIR.Dialect.PDL.operation(
          operands ++
            [
              name: Beaver.MLIR.Attribute.string(op_name),
              attributeNames: Beaver.MLIR.Attribute.array([]),
              operand_segment_sizes:
                Beaver.MLIR.ODS.operand_segment_sizes([
                  length(operands),
                  length(attributes),
                  # in replacement length of results should always be 0
                  0
                ])
            ]
        ) >>> ~t{!pdl.operation}

      Beaver.MLIR.Dialect.PDL.replace([
        beaver_gen_root,
        repl,
        operand_segment_sizes: Beaver.MLIR.ODS.operand_segment_sizes([1, 1, 0])
      ]) >>> []
    end
  end

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
            unquote(rewrite_block_ast)
          end
        end
      end
    end
  end
end
