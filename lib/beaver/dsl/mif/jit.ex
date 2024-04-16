defmodule Beaver.MIF.JIT do
  alias Beaver.MLIR

  defp jit_of_mod(m) do
    import Beaver.MLIR.Conversion

    m
    |> MLIR.Operation.verify!(debug: true)
    |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
    |> convert_scf_to_cf
    |> convert_arith_to_llvm()
    |> convert_index_to_llvm()
    |> convert_func_to_llvm()
    |> MLIR.Pass.Composer.append("finalize-memref-to-llvm")
    |> reconcile_unrealized_casts
    |> MLIR.Pass.Composer.run!(print: System.get_env("DEFM_PRINT_IR") == "1")
    |> MLIR.ExecutionEngine.create!(opt_level: 3, object_dump: true)
    |> tap(&Beaver.MLIR.CAPI.beaver_raw_jit_register_enif(&1.ref))
  end

  def init(module, opts \\ [])

  def init(module, opts) when is_atom(module) do
    name = opts[:name] || module
    opts = Keyword.put_new(opts, :name, name)
    init([module], opts)
  end

  def init(modules, opts) do
    ctx = MLIR.Context.create()
    Beaver.Diagnostic.attach(ctx)
    name = opts[:name]

    jit =
      modules
      |> Enum.map(& &1.__ir__())
      |> Enum.map(&MLIR.Module.create(ctx, &1))
      |> MLIR.Module.merge()
      |> jit_of_mod

    case {name, modules} do
      {name, [_]} when not is_nil(name) ->
        Agent.start_link(fn -> %{ctx: ctx, jit: jit} end, name: name)

      {nil, modules} ->
        for {module, index} <- Enum.with_index(modules) do
          Agent.start_link(fn -> %{ctx: ctx, jit: jit, owner: index == 0} end, name: module)
        end
    end
  end

  def get(module) do
    %{jit: jit} = Agent.get(module, & &1)
    jit
  end

  def invoke(jit, {mod, func, args}) do
    Beaver.MLIR.CAPI.beaver_raw_jit_invoke_with_terms(
      jit.ref,
      to_string(Beaver.MIF.mangling(mod, func)),
      args
    )
  end

  def destroy(module) do
    with %{ctx: ctx, jit: jit, owner: true} <- Agent.get(module, & &1) do
      MLIR.ExecutionEngine.destroy(jit)
      MLIR.Context.destroy(ctx)
    end

    Agent.stop(module)
  end
end
