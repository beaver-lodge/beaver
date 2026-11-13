defmodule TestTOSAPatterns do
  @moduledoc false
  use Beaver
  alias Beaver.MLIR
  alias MLIR.{Attribute, Type}
  alias MLIR.Dialect.{Func, TOSA}
  require Func

  def gen_ir_module(ctx) do
    mlir ctx: ctx do
      module do
        Func.func test_multi_broadcast(
                    function_type:
                      Type.function(
                        [Type.ranked_tensor([1, 3], Type.f32()), Type.ranked_tensor([2, 1], Type.f32())],
                        [Type.ranked_tensor([2, 3], Type.f32())]
                      )
                  ) do
          region do
            block _entry(a >>> Type.ranked_tensor([1, 3], Type.f32()), b >>> Type.ranked_tensor([2, 1], Type.f32())) do
              res = TOSA.add(a, b, one: Attribute.integer(Type.i32(), 1)) >>> Type.ranked_tensor([2, 3], Type.f32())

              res1 = TOSA.add(res, b) >>> Type.ranked_tensor([2, 3], Type.f32())
              Func.return(res1) >>> []
            end
          end
        end
      end
    end
  end

  import Beaver.Pattern

  defpat replace_add_op(benefit: 10) do
    a = value()
    b = value()
    res_t = type()
    {op, _} = TOSA.add(a, b) >>> {:op, [res_t]}

    rewrite op do
      r = TOSA.sub(a, b) >>> res_t
      replace(op, with: r)
    end
  end

  defpat replace_multi_add_op() do
    one = attribute()
    _x = %Range{first: 1, last: 10, step: 2}
    ty = Type.ranked_tensor([2, 3], Type.f32())
    a = value()
    b = value()
    res = TOSA.add(a, b, one: one) >>> ty
    {op, _t} = TOSA.add(res, b) >>> {:op, ty}

    rewrite op do
      types = [Type.ranked_tensor([2, 3], Type.f32())]
      a = TOSA.sub(a, b) >>> types
      a = TOSA.sub(a, b) >>> types
      a = TOSA.sub(a, b) >>> types
      a = TOSA.sub(a, b, one: one) >>> types
      {r, _} = TOSA.sub(a, b) >>> {:op, ty}
      replace(op, with: r)
    end
  end

  for variant <- 1..3 do
    definition =
      quote do
        defpat unquote(:"replace_multi_add_op#{variant}")() do
          one = if unquote(variant) == 1, do: attribute(), else: Attribute.integer(Type.i32(), 1)
          concrete = Type.ranked_tensor([2, 3], Type.f32())
          types = if unquote(variant) == 1, do: concrete, else: [concrete]
          a = value()
          b = value()
          res = TOSA.add(a, b, one: one) >>> types
          ty = if unquote(variant) == 2, do: type(), else: types
          {op, _t} = TOSA.add(res, b) >>> {:op, ty}

          rewrite op do
            replacement_type = if unquote(variant) == 3, do: ty, else: {:op, ty}
            result = TOSA.sub(a, b) >>> replacement_type
            replace(op, with: if(is_tuple(result), do: elem(result, 0), else: result))
          end
        end
      end

    Code.eval_quoted(definition, [], __ENV__)
  end
end
