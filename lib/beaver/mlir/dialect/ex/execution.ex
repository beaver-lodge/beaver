defmodule Beaver.MLIR.Dialect.Ex.Execution do
  @moduledoc "Scheduler, continuation, mailbox-scan and reduction-accounting operations."

  @doc false
  defmacro __using__(_) do
    quote do
      @ex_layer :execution

      defop mailbox_clear(), results: [result: ^term_like]

      # Starts a fresh process table with one initial process and the given capacity.
      defop process_table_reset(cap = ^integer_like), results: [result: ^integer_like]

      # Saves cursor-loop (arg, acc, cursor) state for round-robin resumption.
      defop cont_save(arg = ^integer_like, acc = ^integer_like, cursor = ^integer_like),
        results: [result: ^integer_like]

      # Message arrival invalidates a receive-scan continuation via its epoch.
      defop receive_cont_save(arg = ^integer_like, acc = ^integer_like, cursor = ^integer_like),
        results: [result: ^integer_like]

      defops ~w(
        cont_pending cont_active cont_clear cont_load_arg cont_load_acc cont_load_cursor schedule_next
        mailbox_len monotonic_time receive_start current_entry processes_runnable yield_mark
      )a,
             [], results: [result: ^integer_like]

      defop mailbox_peek(cursor = ^integer_like), results: [result: ^term_like]
      defop mailbox_remove(cursor = ^integer_like), results: [result: ^integer_like]

      defop receive_start_set(value = ^integer_like), results: [result: ^integer_like]

      defop process_done(result = ^integer_like), results: [result: ^integer_like]

      defop process_result(pid = ^term_like), results: [result: ^integer_like]

      # Parks the actor when no message remains beyond its completed scan cursor.
      defop process_wait(cursor = ^integer_like), results: [result: ^integer_like]

      # Fixed worker count; dispatcher is a native `(pid: i64) -> i64` trampoline.
      defop worker_run(worker_count = ^integer_like, dispatcher = ^any_value), results: [result: ^integer_like]

      defop reduction_tick(cost = ^integer_like), results: [result: ^integer_like]
      defop clock_init(budget = ^integer_like), results: [result: ^integer_like]
    end
  end
end
