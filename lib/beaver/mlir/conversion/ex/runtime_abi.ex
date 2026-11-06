defmodule Beaver.MLIR.Conversion.Ex.RuntimeABI do
  @moduledoc "Legacy tagged-word ABI used by the reference Ex lowering; independent of language schemas."
  alias Beaver.MLIR.Type

  # The original Batata term runtime implements this compatibility ABI.
  # The current wavefrontier/batata C runtime has a separate contract.
  @intrinsic_names ~w(
    list_cons list_flatten list_insert_at map_put map_fetch self send receive mailbox_clear spawn runtime_create
    runtime_enter runtime_leave runtime_destroy result_create result_create_term result_destroy result_root_kind
    result_root_word result_exception_kind result_exception_reason result_term_kind result_atom_name
    result_term_length result_term_get term_export term_import exported_clone exported_destroy exported_length
    exported_get term_handle_export term_handle_destroy process_table_reset cont_save receive_cont_save
    cont_pending cont_active cont_clear cont_load_arg cont_load_acc cont_load_cursor schedule_next mailbox_len
    mailbox_peek mailbox_remove nil_word monotonic_time receive_start receive_start_set native_time unique_integer
    current_entry process_done process_exit process_exit_reason process_trap_exit process_dictionary_get
    process_dictionary_put link unlink exit monitor demonitor processes_runnable process_result worker_run
    process_wait to_int reduction_tick clock_init yield_mark jmp_buf_size setjmp_addr try_push try_pop throw raise
    catch_value make_fun make_fun_with_arity make_fun_with_signature fun_arity fun_result_mode fun_idx fun_env
    tuple_from_list map_from_list mapset_from_list mapset_member mapset_put file_read file_read_lines
    binary_from_list iodata_to_binary float_lit bigint_lit string_to_float string_to_atom string_to_existing_atom
    float_to_binary_short is_integer is_float is_atom is_binary is_list is_tuple is_map tuple_get tuple_length
    map_length enumerable_count enumerable_to_list enumerable_into_map enumerable_intersperse
    enumerable_to_list_range enumerable_reduce enumerable_reduce_c enumerable_reduce_range enumerable_reduce_fun
    enumerable_map_fun enumerable_map_term_fun enumerable_map_term_fun_c enumerable_flat_map_term_fun
    stream_filter stream_take stream_drop list_head list_tail list_get list_length term_eq term_eq_loose
    integer_compare integer_add integer_sub integer_mul integer_div integer_rem binary_length binary_get
    binary_slice binary_part binary_utf8_get binary_utf8_width binary_utf8_length string_printable binary_quote
    binary_encode16 binary_decode16 int_to_string int_to_string_base int_to_hex string_to_int
  )
  @intrinsic_aliases %{
    "term_export" => "export",
    "term_import" => "import",
    "term_handle_export" => "handle_export",
    "term_handle_destroy" => "handle_destroy",
    "nil_word" => "nil",
    "reduction_tick" => "clock_tick",
    "term_eq" => "eq",
    "term_eq_loose" => "eq_loose"
  }
  @term_intrinsics Map.new(@intrinsic_names, fn name ->
                     {String.to_atom(name), "ex.term." <> Map.get(@intrinsic_aliases, name, name)}
                   end)
  @symbols Map.values(@term_intrinsics)

  @doc "Operation keys and their runtime symbols."
  def intrinsics, do: @term_intrinsics

  @doc "All symbols required by the reference runtime."
  def symbols, do: Enum.sort(@symbols)

  @doc false
  for {arity, names} <- [
        {0, ~w(
          self receive yield_mark mailbox_clear runtime_create runtime_leave cont_pending cont_active cont_clear
          cont_load_arg cont_load_acc cont_load_cursor schedule_next mailbox_len nil monotonic_time receive_start
          native_time current_entry processes_runnable jmp_buf_size setjmp_addr try_pop catch_value
        )},
        {2, ~w(
          list_cons send raise enumerable_to_list_range fun_env tuple_get list_get eq eq_loose integer_compare
          integer_add integer_sub integer_mul integer_div integer_rem binary_get binary_slice binary_utf8_get
          binary_utf8_width int_to_string_base enumerable_into_map enumerable_intersperse map_fetch mapset_member
          mapset_put stream_take stream_drop result_create result_create_term result_term_kind result_atom_name
          result_term_length export import exported_get process_dictionary_get process_dictionary_put
        )},
        {3, ~w(map_put list_insert_at result_term_get cont_save receive_cont_save link enumerable_reduce binary_part)},
        {4, ~w(exit monitor enumerable_reduce_c enumerable_reduce_range)},
        {6, ~w(make_fun)},
        {7, ~w(make_fun_with_arity)},
        {8, ~w(make_fun_with_signature)}
      ],
      name <- names do
    def function_type(unquote("ex.term." <> name), _ctx),
      do: Type.function(List.duplicate(Type.i64(), unquote(arity)), [Type.i64()])
  end

  def function_type("ex.term.try_push", ctx), do: Type.function([Type.llvm_pointer(ctx: ctx)], [Type.i64()])

  def function_type("ex.term.enumerable_reduce_fun", _ctx),
    do: Type.function([Type.i64(), Type.i64(), Type.function([Type.i64(), Type.i64()], [Type.i64()])], [Type.i64()])

  def function_type("ex.term.worker_run", _ctx),
    do: Type.function([Type.i64(), Type.function([Type.i64()], [Type.i64()])], [Type.i64()])

  def function_type(symbol, _ctx)
      when symbol in ~w(
         ex.term.enumerable_map_fun ex.term.enumerable_map_term_fun ex.term.enumerable_flat_map_term_fun
         ex.term.stream_filter
       ) do
    Type.function([Type.i64(), Type.function([Type.i64()], [Type.i64()])], [Type.i64()])
  end

  def function_type("ex.term.enumerable_map_term_fun_c", _ctx) do
    callback = Type.function(List.duplicate(Type.i64(), 8), [Type.i64()])

    Type.function([Type.i64(), callback | List.duplicate(Type.i64(), 4)], [Type.i64()])
  end

  def function_type(symbol, _ctx) when symbol in @symbols, do: Type.function([Type.i64()], [Type.i64()])
end
