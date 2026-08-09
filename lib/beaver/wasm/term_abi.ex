defmodule Beaver.Wasm.TermABI do
  @moduledoc """
  The WASI host-boundary ABI manifest for the `ex.term.*` term runtime.

  On native, `Beaver.MLIR.Conversion.Ex` emits calls to the `ex.term.*`
  symbols and batata's Zig `term_runtime` implements them (the declaration
  lives in `batata/native/ABI.md`). On wasm the same symbols must be
  provided by a **host import** — there is no linked runtime. This module
  declares that boundary: for every intrinsic, the wasm import module and
  name plus the C signature a host must implement.

  The manifest is declaration-first, mirroring the existing pattern:
  `Conversion.Ex.term_intrinsic_symbols/0` is the symbol anchor (what the
  compiler emits), and this module adds the wasm host contract (what the
  runtime must provide). A wasm host (the future `:wasm` Shadow Wavefront
  evaluator) implements exactly these imports.
  """

  alias Beaver.MLIR.Conversion.Ex

  @import_module "ex_term"

  @type param :: :i64 | :ptr
  @type result :: :i64 | :void

  @type entry :: %{
          symbol: String.t(),
          import_module: String.t(),
          import_name: String.t(),
          params: [param()],
          result: result()
        }

  # The signatures mirror `batata/native/ABI.md` and the function types
  # emitted by `Beaver.MLIR.Conversion.Ex.intrinsic_function_type/2`.
  @intrinsics %{
    # actor / mailbox
    "ex.term.self" => %{params: [], result: :i64},
    "ex.term.send" => %{params: [:i64, :i64], result: :i64},
    "ex.term.receive" => %{params: [], result: :i64},
    "ex.term.mailbox_len" => %{params: [], result: :i64},
    "ex.term.mailbox_peek" => %{params: [:i64], result: :i64},
    "ex.term.mailbox_remove" => %{params: [:i64], result: :i64},
    "ex.term.nil" => %{params: [], result: :i64},
    "ex.term.mailbox_clear" => %{params: [], result: :i64},
    # process table / scheduler driver
    "ex.term.spawn" => %{params: [:i64], result: :i64},
    "ex.term.process_table_reset" => %{params: [], result: :i64},
    "ex.term.schedule_next" => %{params: [], result: :i64},
    "ex.term.current_entry" => %{params: [], result: :i64},
    "ex.term.process_done" => %{params: [:i64], result: :i64},
    "ex.term.processes_runnable" => %{params: [], result: :i64},
    "ex.term.process_result" => %{params: [:i64], result: :i64},
    # continuations
    "ex.term.cont_save" => %{params: [:i64, :i64, :i64], result: :i64},
    "ex.term.receive_cont_save" => %{params: [:i64, :i64, :i64], result: :i64},
    "ex.term.cont_pending" => %{params: [], result: :i64},
    "ex.term.cont_active" => %{params: [], result: :i64},
    "ex.term.cont_clear" => %{params: [], result: :i64},
    "ex.term.cont_load_arg" => %{params: [], result: :i64},
    "ex.term.cont_load_acc" => %{params: [], result: :i64},
    "ex.term.cont_load_cursor" => %{params: [], result: :i64},
    # preemption / continuation clock
    "ex.term.clock_init" => %{params: [:i64], result: :i64},
    "ex.term.clock_tick" => %{params: [:i64], result: :i64},
    "ex.term.yield_mark" => %{params: [], result: :i64},
    # term construction / inspection
    "ex.term.to_int" => %{params: [:i64], result: :i64},
    "ex.term.eq" => %{params: [:i64, :i64], result: :i64},
    "ex.term.is_integer" => %{params: [:i64], result: :i64},
    "ex.term.is_atom" => %{params: [:i64], result: :i64},
    "ex.term.is_binary" => %{params: [:i64], result: :i64},
    "ex.term.is_list" => %{params: [:i64], result: :i64},
    "ex.term.is_tuple" => %{params: [:i64], result: :i64},
    "ex.term.is_map" => %{params: [:i64], result: :i64},
    "ex.term.list_cons" => %{params: [:i64, :i64], result: :i64},
    "ex.term.list_head" => %{params: [:i64], result: :i64},
    "ex.term.list_tail" => %{params: [:i64], result: :i64},
    "ex.term.list_get" => %{params: [:i64, :i64], result: :i64},
    "ex.term.list_length" => %{params: [:i64], result: :i64},
    "ex.term.tuple_from_list" => %{params: [:i64], result: :i64},
    "ex.term.tuple_get" => %{params: [:i64, :i64], result: :i64},
    "ex.term.tuple_length" => %{params: [:i64], result: :i64},
    "ex.term.map_from_list" => %{params: [:i64], result: :i64},
    "ex.term.map_length" => %{params: [:i64], result: :i64},
    "ex.term.mapset_from_list" => %{params: [:i64], result: :i64},
    "ex.term.mapset_member" => %{params: [:i64, :i64], result: :i64},
    "ex.term.mapset_put" => %{params: [:i64, :i64], result: :i64},
    # binary
    "ex.term.binary_from_list" => %{params: [:i64], result: :i64},
    "ex.term.binary_length" => %{params: [:i64], result: :i64},
    "ex.term.binary_get" => %{params: [:i64, :i64], result: :i64},
    "ex.term.binary_slice" => %{params: [:i64, :i64], result: :i64},
    "ex.term.binary_utf8_get" => %{params: [:i64, :i64], result: :i64},
    "ex.term.binary_utf8_width" => %{params: [:i64, :i64], result: :i64},
    "ex.term.binary_utf8_length" => %{params: [:i64], result: :i64},
    "ex.term.binary_encode16" => %{params: [:i64], result: :i64},
    "ex.term.binary_decode16" => %{params: [:i64], result: :i64},
    "ex.term.int_to_string" => %{params: [:i64], result: :i64},
    "ex.term.string_to_int" => %{params: [:i64], result: :i64},
    # file IO
    "ex.term.file_read" => %{params: [:i64], result: :i64},
    "ex.term.file_read_lines" => %{params: [:i64], result: :i64},
    # enumerable / stream
    "ex.term.enumerable_count" => %{params: [:i64], result: :i64},
    "ex.term.enumerable_to_list" => %{params: [:i64], result: :i64},
    "ex.term.enumerable_to_list_range" => %{params: [:i64, :i64], result: :i64},
    "ex.term.enumerable_reduce" => %{params: [:i64, :i64, :i64], result: :i64},
    "ex.term.enumerable_reduce_c" => %{params: [:i64, :i64, :i64, :i64], result: :i64},
    "ex.term.enumerable_reduce_range" => %{params: [:i64, :i64, :i64, :i64], result: :i64},
    "ex.term.enumerable_reduce_fun" => %{params: [:i64, :i64, :i64], result: :i64},
    "ex.term.enumerable_map_fun" => %{params: [:i64, :i64], result: :i64},
    "ex.term.stream_filter" => %{params: [:i64, :i64], result: :i64},
    "ex.term.stream_take" => %{params: [:i64, :i64], result: :i64},
    "ex.term.stream_drop" => %{params: [:i64, :i64], result: :i64},
    # closures / callbacks
    "ex.term.make_fun" => %{
      params: [:i64, :i64, :i64, :i64, :i64, :i64],
      result: :i64
    },
    "ex.term.fun_idx" => %{params: [:i64], result: :i64},
    "ex.term.fun_env" => %{params: [:i64, :i64], result: :i64},
    # exception / setjmp
    "ex.term.jmp_buf_size" => %{params: [], result: :i64},
    "ex.term.setjmp_addr" => %{params: [], result: :i64},
    "ex.term.try_push" => %{params: [:ptr], result: :i64},
    "ex.term.try_pop" => %{params: [], result: :i64},
    "ex.term.throw" => %{params: [:i64], result: :void},
    "ex.term.catch_value" => %{params: [], result: :i64}
  }

  @doc """
  Returns the wasm host-boundary manifest: one entry per term intrinsic.

  `import_module`/`import_name` are the wasm import the host must provide;
  `params`/`result` are the C ABI signature (`:i64` tagged word, `:ptr`
  host pointer, `:void` for noreturn).
  """
  @spec manifest() :: [entry()]
  def manifest do
    Enum.map(@intrinsics, fn {symbol, signature} ->
      Map.merge(signature, %{
        symbol: symbol,
        import_module: @import_module,
        import_name: symbol
      })
    end)
  end

  @doc "Looks up the host-boundary entry for an `ex.term.*` symbol."
  @spec entry(String.t()) :: entry() | nil
  def entry(symbol) when is_binary(symbol) do
    case Map.fetch(@intrinsics, symbol) do
      {:ok, signature} ->
        Map.merge(signature, %{
          symbol: symbol,
          import_module: @import_module,
          import_name: symbol
        })

      :error ->
        nil
    end
  end

  @doc "All `ex.term.*` symbols declared by the manifest."
  @spec symbols() :: [String.t()]
  def symbols, do: @intrinsics |> Map.keys() |> Enum.sort()

  @doc """
  Renders the manifest as a Markdown table (paste-ready for an upstream
  issue or the wasm host documentation).
  """
  @spec render() :: String.t()
  def render do
    rows =
      Enum.map_join(manifest(), "\n", fn entry ->
        params = Enum.map_join(entry.params, ", ", &format_type/1)
        result = if entry.result == :void, do: "noreturn", else: format_type(entry.result)
        "| `#{entry.symbol}` | `#{entry.import_module}` | `(#{params}) -> #{result}` |"
      end)

    """
    ## ex.term.* wasm host imports

    | symbol | import module | signature |
    | --- | --- | --- |
    #{rows}
    """
  end

  @doc false
  @spec coverage() :: %{missing: [String.t()], extra: [String.t()]}
  def coverage do
    emitted = Ex.term_intrinsic_symbols()
    declared = symbols()

    %{
      missing: emitted -- declared,
      extra: declared -- emitted
    }
  end

  defp format_type(:i64), do: "i64"
  defp format_type(:ptr), do: "ptr"
end
