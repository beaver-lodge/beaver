defmodule Beaver.DSL.Pattern do
  use Beaver

  defp do_transform_match(
         {:%, _,
          [
            struct_name,
            {:%{}, _, map_args}
          ]} = _ast
       ) do
    operands = map_args |> Keyword.get(:operands, [])
    results = map_args |> Keyword.get(:results, [])
    _attributes = map_args |> Keyword.get(:attributes, [])

    filtered_operands =
      Enum.map(operands, fn
        {:bound, bound} -> bound
        other -> other
      end)

    map_args = Keyword.put(map_args, :operands, filtered_operands)

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
        unquote_splicing(results_match)

        beaver_gen_root =
          Beaver.DSL.Op.Prototype.dispatch(
            unquote(struct_name),
            unquote(map_args),
            &Beaver.DSL.Pattern.create_root/2
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
          &Beaver.DSL.Pattern.create_rewrite/3
        )
      end

    ast
  end

  defp do_transform_rewrite(ast), do: ast

  def create_root(
        op_name,
        %Beaver.DSL.Op.Prototype{operands: operands, attributes: attributes, results: results}
      ) do
    mlir do
      arguments = operands ++ attributes ++ results

      Beaver.MLIR.Dialect.PDL.operation(
        arguments ++
          [
            name: Beaver.MLIR.Attribute.string(op_name),
            attributeNames: Beaver.MLIR.Attribute.array([]),
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

  def create_rewrite(
        op_name,
        %Beaver.DSL.Op.Prototype{operands: operands, attributes: attributes} = prototype,
        beaver_gen_root
      ) do
    mlir do
      Beaver.MLIR.Dialect.PDL.rewrite [
        beaver_gen_root,
        defer_if_terminator: false,
        operand_segment_sizes: Beaver.MLIR.ODS.operand_segment_sizes([1, 0])
      ] do
        region do
          block some() do
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
      end
    end
  end

  def transform_rewrite(ast) do
    Macro.postwalk(ast, &do_transform_rewrite/1)
  end
end
