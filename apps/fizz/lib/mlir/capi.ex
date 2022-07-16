defmodule Fizz.MLIR.CAPI do
  wrapper = Path.join(File.cwd!(), "capi/src/llvm-15.h")
  {functions, types} = Fizz.MLIR.NIFGen.gen(wrapper)
  @on_load :load_nifs

  def load_nifs do
    :erlang.load_nif('capi/zig-out/lib/libBeaverCAPI', 0)
  end

  for type <- types do
    defmodule Module.concat(__MODULE__, Fizz.MLIR.NIFGen.module_name(type)) do
      defstruct ref: nil, zig_t: String.to_atom(type)
    end
  end

  for f <- Path.wildcard("capi/src") do
    @external_resource f
  end

  @external_resource "capi/build.zig"
  for %Fizz.CodeGen.Function{name: name, args: args, ret: ret} <- functions do
    arity = length(args)

    args_ast = Macro.generate_unique_arguments(arity, __MODULE__)

    fizz_func_name = String.to_atom("fizz_nif_" <> name)

    return_module = Module.concat(__MODULE__, Fizz.MLIR.NIFGen.module_name(ret))

    def unquote(String.to_atom(name))(unquote_splicing(args_ast)) do
      refs =
        [unquote_splicing(args_ast)]
        |> Enum.map(fn %{ref: ref} -> ref end)

      ref = apply(__MODULE__, unquote(fizz_func_name), refs)
      struct!(unquote(return_module), %{ref: ref, zig_t: String.to_atom(unquote(ret))})
    end

    def unquote(fizz_func_name)(unquote_splicing(args_ast)),
      do: "failed to load NIF"
  end

  def test() do
  end
end
