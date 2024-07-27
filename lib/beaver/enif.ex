defmodule Beaver.ENIF do
  @moduledoc """
  This module provides functions to work with Erlang's [erl_nif](https://www.erlang.org/doc/man/erl_nif.html) APIs in MLIR.

  ## Main usages
  - call `declare_external_functions/2` to insert external function declarations into a `Beaver.MLIR.Block`
  - call `register_symbols/1` to register symbols of ENIF functions in a `Beaver.MLIR.ExecutionEngine`
  """

  use Beaver
  require Beaver.MLIR.Dialect.Func
  alias Beaver.MLIR.Dialect.Func
  alias MLIR.{Type, Attribute}

  defp wrap_mlir_t({ref, _size}) when is_reference(ref) do
    %MLIR.Type{ref: ref}
  end

  @doc """
  Insert external functions of ENIF into given MLIR block
  """
  def declare_external_functions(ctx, block) do
    mlir ctx: ctx, block: block do
      for {name, arg_types, ret_types} <- signatures(ctx) do
        Func.func _(
                    sym_name: "\"#{name}\"",
                    sym_visibility: MLIR.Attribute.string("private"),
                    function_type: Type.function(arg_types, ret_types)
                  ) do
          region do
          end
        end
      end
    end
  end

  @spec signatures(MLIR.Context.t()) :: [Beaver.ENIF.Type.signature()]
  @doc """
  Retrieve the signatures of all available ENIF functions.
  """
  def signatures(%MLIR.Context{} = ctx) do
    signatures = MLIR.CAPI.beaver_raw_enif_signatures(ctx.ref)

    for {name, arg_types, ret_types} <- signatures do
      {name, Enum.map(arg_types, &wrap_mlir_t/1), Enum.map(ret_types, &wrap_mlir_t/1)}
    end
  end

  @spec signature(MLIR.Context.t(), atom()) :: nil | Beaver.ENIF.Type.signature()
  @doc """
  Query the signature of an ENIF function.
  """
  def signature(%MLIR.Context{} = ctx, name) do
    for {^name, arg_types, ret_types} <- signatures(ctx) do
      {arg_types, ret_types}
    end
    |> List.first()
  end

  # Insert a function call to an ENIF function into a MLIR block with appropriate return type.
  defp call(f, %Beaver.SSA{arguments: arguments, block: block, ctx: ctx, loc: loc}) do
    mlir block: block, ctx: ctx do
      {_, t} = Beaver.ENIF.signature(Beaver.Env.context(), f)
      symbol = Attribute.flat_symbol_ref(f)
      Func.call(arguments, callee: symbol, loc: loc) >>> t
    end
  end

  for f <- MLIR.CAPI.beaver_raw_enif_functions() do
    case to_string(f) do
      "enif_" <> op ->
        @doc """
        function call to [#{f}](https://www.erlang.org/doc/apps/erts/erl_nif.html##{f})
        """
        def unquote(String.to_atom(op))(ssa) do
          call(unquote(f), ssa)
        end

      _ ->
        @doc (case f do
                :ptr_to_memref ->
                  """
                  Convert a pointer to a memref

                  `ptr_to_memref(ptr(), size()) :: memref<?xi8>`
                  """

                _ ->
                  """
                  function call to #{f}
                  """
              end)
        def unquote(f)(ssa) do
          call(unquote(f), ssa)
        end
    end
  end

  defdelegate functions(), to: MLIR.CAPI, as: :beaver_raw_enif_functions

  def register_symbols(%MLIR.ExecutionEngine{ref: ref} = e) do
    MLIR.CAPI.beaver_raw_jit_register_enif(ref)
    e
  end

  @spec invoke(
          MLIR.ExecutionEngine.t(),
          String.t(),
          list(),
          MLIR.ExecutionEngine.invoke_opts()
        ) ::
          term()
  def invoke(%MLIR.ExecutionEngine{ref: ref}, function, arguments, opts \\ []) do
    case opts[:dirty] do
      :cpu_bound ->
        :beaver_raw_jit_invoke_with_terms_cpu_bound

      :io_bound ->
        :beaver_raw_jit_invoke_with_terms_io_bound

      nil ->
        :beaver_raw_jit_invoke_with_terms
    end
    |> then(&apply(MLIR.CAPI, &1, [ref, function, arguments]))
  end
end
