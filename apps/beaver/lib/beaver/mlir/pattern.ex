defmodule Beaver.MLIR.Pattern do
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Although this module is `MLIR.Pattern`, at this point it is a synonym of PDL patterns.
  Pattern-matching is done by MLIR which works in a different way from Erlang pattern-matching.
  The major difference is that MLIR pattern-matching will greedily match the patterns and maximize the benifit.
  Compiled patterns will be saved as module attributes in MLIR assembly format.
  """
  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__), only: [pattern: 2]
      Module.register_attribute(__MODULE__, :compiled_pattern, accumulate: true, persist: true)
    end
  end

  defmacro pattern(call, do: block) do
    {name, args} = Macro.decompose_call(call)

    alias Beaver.MLIR.Dialect.PDL
    alias Beaver.MLIR.Attribute
    alias Beaver.MLIR.Type

    pdl_pattern_op =
      mlir do
        module do
          PDL.pattern benefit: Attribute.integer(Type.i16(), 1) do
            region do
              block some_pattern() do
                t = PDL.type() >>> ~t{!pdl.type}
                a = PDL.operand() >>> ~t{!pdl.value}
                b = PDL.operand() >>> ~t{!pdl.value}

                root =
                  PDL.operation(a, b, t,
                    name: Attribute.string("tosa.add"),
                    attributeNames: Attribute.array([]),
                    operand_segment_sizes: ODS.operand_segment_sizes([2, 0, 1])
                  ) >>> ~t{!pdl.operation}

                PDL.rewrite [
                  root,
                  defer_if_terminator: false,
                  operand_segment_sizes: ODS.operand_segment_sizes([1, 0])
                ] do
                  region do
                    block some() do
                      repl =
                        PDL.operation(a, b,
                          name: Attribute.string("tosa.sub"),
                          attributeNames: Attribute.array([]),
                          operand_segment_sizes: ODS.operand_segment_sizes([2, 0, 0])
                        ) >>> ~t{!pdl.operation}

                      PDL.replace([
                        root,
                        repl,
                        operand_segment_sizes: ODS.operand_segment_sizes([1, 1, 0])
                      ]) >>> []

                      # PDL.erase(root) >>> []
                    end
                  end
                end
              end
            end
          end
        end
      end

    pattern_string = MLIR.Operation.to_string(pdl_pattern_op) |> String.trim()

    MLIR.Operation.verify!(pdl_pattern_op, dump: true)

    quote do
      @compiled_pattern unquote(pattern_string)
    end
  end

  def compiled_patterns(module) when is_atom(module) do
    apply(module, :__info__, [:attributes])[:compiled_pattern]
  end

  def from_string(ctx, pdl_pattern_str) when is_binary(pdl_pattern_str) do
    pattern_module = ~m{#{pdl_pattern_str}}
    if MLIR.Module.is_null(pattern_module), do: raise("fail to parse module")
    MLIR.Operation.verify!(pattern_module)
    pdl_pattern = CAPI.beaverPDLPatternGet(pattern_module)
    pdl_pattern
  end
end
