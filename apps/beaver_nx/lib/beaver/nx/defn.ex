defmodule Beaver.Nx.Defn do
  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  require Beaver
  import Beaver, only: [mlir: 1]
  require Beaver.MLIR.Dialect.Func
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.{Builtin, Func, TOSA, Arith}
  import Builtin, only: :macros
  import MLIR, only: :macros
  import MLIR.Sigils

  defp gen_tensor_type_str(%Nx.Tensor{shape: {}, type: {:s, size}}) do
    "tensor<i#{size}>"
  end

  def gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}})
      when is_integer(pos) do
    MLIR.Managed.Block.get()
    |> Beaver.MLIR.Block.get_arg!(pos)
  end

  def gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :multiply, args: [a, b]} = expr}) do
    mlir do
      a = gen_op(a)
      b = gen_op(b)
      # TOSA.mul(a, b, {:shift, ~a{0 : i32}}) :: ~t{tensor<i64>}
    end

    IO.inspect(expr, label: "to add", structs: false)
  end

  @doc false
  def __jit__(key, vars, fun, args, options) do
    exprs = fun.(vars) |> IO.inspect(label: "exprs")
    IO.inspect(vars, label: "vars")
    # IO.inspect({key, vars, fun, args, options})

    arg_types_str =
      for arg <- vars do
        IO.inspect(arg, label: "arg", structs: false)
        gen_tensor_type_str(arg)
      end
      |> Enum.join(", ")

    t = "tensor<i64>"

    ir =
      mlir do
        module do
          Func.func main(function_type: ~a"(#{arg_types_str}) -> ()") do
            region do
              block entry(arg0 :: ~t{#{t}}, arg1 :: ~t{#{t}}) do
                v0 = TOSA.mul(arg0, arg1, {:shift, ~a{0 : i32}}) :: ~t{#{t}}

                Composite.reduce(exprs, [], fn %T{} = expr, acc ->
                  # expr = gen_op(expr)
                  [expr | acc]
                end)

                Func.return([])
              end
            end
          end
        end
      end
      |> MLIR.Operation.dump!()

    t = Nx.tensor(121) |> Beaver.Nx.from_binary(<<121::64-native>>, [])
    [t]
  end
end
