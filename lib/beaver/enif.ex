defmodule Beaver.ENIF do
  @erlang_doc_enif_url "https://www.erlang.org/doc/apps/erts/erl_nif.html"
  @make_new_binary_as_memref "call [`enif_make_new_binary`](#{@erlang_doc_enif_url}#enif_make_new_binary) and save the data pointer and size to a `memref<?xi8>`"
  @inspect_binary_as_memref "call [`enif_inspect_binary`](#{@erlang_doc_enif_url}#enif_inspect_binary) and save the data pointer and size to a `memref<?xi8>`"
  @moduledoc """
  This module provides functions to work with Erlang's [erl_nif](https://www.erlang.org/doc/man/erl_nif.html) APIs in MLIR.

  ## Main usages
  - call `declare_external_functions/2` to insert external function declarations into a `Beaver.MLIR.Block`
  - call `register_symbols/1` to register symbols of ENIF functions in a `Beaver.MLIR.ExecutionEngine`

  ## Supplemental Functions for ENIF and MLIR Runtime Interoperation

  Beaver comes with supplemental functions to facilitate the interoperation between ENIF and MLIR runtime:

  ### Rationale
  - Arguments (including value and pointer) in the supplement functions should be created by MLIR
  - Avoid reinterpreting a raw pointer created by ENIF or other C functions to a MLIR representation,
  this ensures the stability of the ABI and clarity of semantics.
  For instance, we should not generate IR to reinterpret a binary buffer to `memref<?xi8>` directly.
  Instead, we provide a function `inspect_binary_as_memref/1` to perform the conversion explicitly.
  - These functions and existing ENIF functions will be defined as an MLIR dialect in the future once IRDL is mature enough.

  ### Provided Functions
  - `make_new_binary_as_memref/1`: #{@make_new_binary_as_memref}
  - `inspect_binary_as_memref/1`: #{@inspect_binary_as_memref}
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
    mlir ctx: ctx, blk: block do
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
  def signatures(%MLIR.Context{ref: ref}) do
    signatures = MLIR.CAPI.beaver_raw_enif_signatures(ref)

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
  defp call(f, %Beaver.SSA{arguments: arguments, blk: block, ctx: ctx, loc: loc}) do
    mlir blk: block, ctx: ctx do
      {_, t} = Beaver.ENIF.signature(Beaver.Env.context(), f)
      symbol = Attribute.flat_symbol_ref(f)
      Func.call(arguments, callee: symbol, loc: loc) >>> t
    end
  end

  for f <- MLIR.CAPI.beaver_raw_enif_functions() do
    case to_string(f) do
      "enif_" <> op ->
        @doc """
        function call to [`#{f}`](#{@erlang_doc_enif_url}##{f})
        """
        def unquote(String.to_atom(op))(ssa) do
          call(unquote(f), ssa)
        end

      _ ->
        @doc (case f do
                :make_new_binary_as_memref ->
                  """
                  #{@make_new_binary_as_memref}
                  """

                :inspect_binary_as_memref ->
                  """
                  #{@inspect_binary_as_memref}
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

  @doc """
  Register ENIF functions in the given `Beaver.MLIR.ExecutionEngine`.
  """
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
  @doc """
  Invoke a function in the given `Beaver.MLIR.ExecutionEngine` with arguments and return all have type `Beaver.ENIF.Type.term/1`.
  """
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
