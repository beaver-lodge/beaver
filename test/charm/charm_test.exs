defmodule Beaver.Charm.Struct do
  use Beaver
  defstruct field_types: [], ctx: nil

  defmacro __using__(fields) do
    quote do
      def __charmtype__ do
        unquote(fields)
      end
    end
  end

  def update_field(%Beaver.Charm.Struct{ctx: ctx, field_types: field_types}, field, cb) do
    mlir ctx: ctx do
      module do
        cb.(ctx, Beaver.Env.block(), field_types[field])
      end
    end
    |> MLIR.Operation.verify!()
  end
end

defmodule Beaver.Charm do
  use Beaver

  defmodule CompilerCtx do
    @moduledoc false
    defstruct ctx: nil
  end

  defp gen_mlir(ssa, acc) do
    ssa |> dbg
    {ssa, acc}
  end

  def compile(ssa) do
    ctx = MLIR.Context.create()
    Intermediator.SSA.peek(ssa)
    Intermediator.SSA.prewalk(ssa, %CompilerCtx{ctx: ctx}, &gen_mlir/2)
  end

  def run(mod) do
  end
end

defmodule Charm.Boolean do
  use Beaver
  alias MLIR.Type
  alias Beaver.Charm
  alias MLIR.Dialect.Index
  alias Beaver.Charm.Struct
  use Charm.Struct, value: Type.i1()

  defstruct value: false

  def __init__(%Beaver.Charm.Struct{ctx: ctx} = this) do
    Struct.update_field(this, :value, fn ctx, block, type ->
      mlir block: block, ctx: ctx do
        Index.bool_constant(value: ~a{false}) >>> type
      end
    end)
  end

  def __init__() do
    ctx = MLIR.Context.create()
    this = %Beaver.Charm.Struct{field_types: __charmtype__(), ctx: ctx}
    __init__(this)
  end
end

defmodule CharmTest do
  @moduledoc false
  use ExUnit.Case
  use Beaver

  defmodule CharmTest do
  end

  test "valid syntax" do
    %{MyBoolean => ssa} =
      File.cwd!()
      |> Path.join("test/charm/examples/boolean.ex")
      |> List.wrap()
      |> Intermediator.SSA.compile()

    ssa
    |> Beaver.Charm.compile()
    |> Beaver.Charm.run()
  end
end
