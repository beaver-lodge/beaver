alias Beaver.MLIR.Attribute
alias Beaver.Deferred

defmodule Beaver.MLIR.Dialect.Linalg do
  @moduledoc """
  Operations and structured-compute builders for the MLIR Linalg dialect.
  """

  alias Beaver.MLIR

  use Beaver.MLIR.Dialect, dialect: "linalg", ops: Beaver.MLIR.Dialect.Registry.ops("linalg")

  @doc """
  Build a `linalg.generic` with structured inputs, outputs, maps, iterators,
  scalar block arguments, and an implicit `linalg.yield`.

  The body receives two lists: scalar input values and scalar output values.
  Its return value is yielded.

      Linalg.generic(
        inputs: [lhs, rhs],
        outputs: [init],
        indexing_maps: [identity, identity, identity],
        iterators: [:parallel]
      ) do
        fn [lhs, rhs], [_out] -> Arith.addf(lhs, rhs) >>> Type.f32() end
      end
  """
  defmacro generic(opts, do: body) do
    inputs = Keyword.fetch!(opts, :inputs)
    outputs = Keyword.fetch!(opts, :outputs)
    maps = Keyword.fetch!(opts, :indexing_maps)
    iterators = Keyword.fetch!(opts, :iterators)

    unless is_list(inputs) and is_list(outputs) and is_list(maps) and is_list(iterators),
      do: raise(ArgumentError, "linalg.generic inputs, outputs, indexing_maps, and iterators must be lists")

    input_vars = scalar_vars(:linalg_input, length(inputs))
    output_vars = scalar_vars(:linalg_output, length(outputs))

    block_args =
      Enum.zip_with(input_vars ++ output_vars, inputs ++ outputs, fn variable, shaped ->
        quote do
          unquote(variable) >>> Beaver.MLIR.Dialect.Linalg.block_argument_type(unquote(shaped))
        end
      end)

    block_call = {:_linalg_body, [], block_args}

    quote do
      Beaver.mlir do
        Beaver.MLIR.Dialect.Linalg.generic inputs: unquote(inputs),
                                           outputs: unquote(outputs),
                                           indexing_maps: Beaver.MLIR.Dialect.Linalg.indexing_maps(unquote(maps)),
                                           iterator_types:
                                             Beaver.MLIR.Dialect.Linalg.iterator_types(unquote(iterators)),
                                           operand_segment_sizes: :infer do
          region do
            block unquote(block_call) do
              Beaver.MLIR.Dialect.Linalg.yield(unquote(body).(unquote(input_vars), unquote(output_vars))) >>> []
            end
          end
        end >>> Beaver.MLIR.Dialect.Linalg.result_types(unquote(outputs))
      end
    end
  end

  @doc "Build the array attribute used by `linalg.generic` indexing maps."
  def indexing_maps(maps) when is_list(maps) do
    Deferred.defer(fn ctx ->
      maps
      |> Enum.map(fn map -> map |> Deferred.resolve(ctx) |> Attribute.affine_map() end)
      |> Attribute.array(ctx: ctx)
    end)
  end

  @doc "Build and validate Linalg iterator types."
  def iterator_types(iterators) when is_list(iterators) do
    allowed = [:parallel, :reduction, :window]

    case iterators -- allowed do
      [] -> :ok
      [invalid | _] -> raise ArgumentError, "unsupported Linalg iterator type: #{inspect(invalid)}"
    end

    Deferred.defer(fn ctx ->
      iterators |> Enum.map(&Attribute.get("#linalg.iterator_type<#{&1}>", ctx: ctx)) |> Attribute.array(ctx: ctx)
    end)
  end

  @doc false
  def block_argument_type(value) do
    type = MLIR.Value.type(value)
    if MLIR.Type.shaped?(type), do: MLIR.ShapedType.element_type(type), else: type
  end

  @doc false
  def result_types(outputs),
    do: outputs |> Enum.map(&MLIR.Value.type/1) |> Enum.filter(&MLIR.Type.tensor?/1)

  defp scalar_vars(prefix, count) do
    for index <- 0..(count - 1)//1, do: Macro.var(:"#{prefix}_#{index}", nil)
  end
end
