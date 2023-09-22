defmodule IRDLTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke

  test "gen irdl", test_context do
    import Beaver.MLIR.Sigils
    import MLIR.{Transforms, Conversion}

    arg = Beaver.Native.I32.make(42)
    return = Beaver.Native.I32.make(-1)

    ctx = test_context[:ctx]

    ~m"""
    irdl.dialect @cmath {
      irdl.type @complex {
        %0 = irdl.is f32
        %1 = irdl.is f64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }
      irdl.operation @norm {
        %0 = irdl.any
        %1 = irdl.parametric @complex<%0>
        irdl.operands(%1)
        irdl.results(%0)
      }
      irdl.operation @mul {
        %0 = irdl.is f32
        %1 = irdl.is f64
        %2 = irdl.any_of(%0, %1)
        %3 = irdl.parametric @complex<%2>
        irdl.operands(%3, %3)
        irdl.results(%3)
      }
    }
    """.(ctx)
    |> MLIR.Operation.verify!()
    |> MLIR.dump!(generic: false)
  end

  test "define dialect" do
    defmodule CMath.Types do
      alias Beaver.MLIR.Type
      import Beaver.Slang

      any_f = any_of([Type.f32(), Type.f64()])

      deftype complex(t = ^any_f) do
      end
    end

    defmodule CMath do
      import CMath.Types
      use Beaver.Slang, name: "cmath"
      c_any = complex(any())

      defop norm(t = ^c_any) do
        any()
      end

      defop mul(c = ^c_any, c) do
        c
      end
    end
  end
end
