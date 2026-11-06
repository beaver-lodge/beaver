defmodule Beaver.MLIR.Dialect.Ex.Execution do
  @moduledoc "Scheduler, continuation, mailbox-scan and reduction-accounting operations."

  @doc false
  defmacro __using__(_) do
    quote do
      @ex_layer :execution

      defop mailbox_clear(), results: [result: ^term_like]

      # Resets the runtime process table to a single fresh initial process with
      # the given capacity; the scheduler driver calls this at program start.
      defop process_table_reset(cap = ^integer_like), results: [result: ^integer_like]

      # Preemptive scheduler continuation primitives (#35 slice 5): a budgeted
      # cursor loop saves its (arg, acc, cursor) state before yielding; the
      # scheduler driver round-robins runnable processes and resumes the saved
      # continuation. A message arrival bumps the recipient's epoch, so a stale
      # continuation reads as not pending (restart from the top).
      defop cont_save(arg = ^integer_like, acc = ^integer_like, cursor = ^integer_like),
        results: [result: ^integer_like]

      # Selective-receive scan continuation: unlike a cursor-loop continuation, a
      # message arrival invalidates it so the scan restarts (epoch wiring).
      defop receive_cont_save(arg = ^integer_like, acc = ^integer_like, cursor = ^integer_like),
        results: [result: ^integer_like]

      defop cont_pending(), results: [result: ^integer_like]

      defop cont_active(), results: [result: ^integer_like]

      defop cont_clear(), results: [result: ^integer_like]

      defop cont_load_arg(), results: [result: ^integer_like]

      defop cont_load_acc(), results: [result: ^integer_like]

      defop cont_load_cursor(), results: [result: ^integer_like]

      defop schedule_next(), results: [result: ^integer_like]

      defop mailbox_len(), results: [result: ^integer_like]

      defop mailbox_peek(cursor = ^integer_like), results: [result: ^term_like]

      defop mailbox_remove(cursor = ^integer_like), results: [result: ^integer_like]

      defop monotonic_time(), results: [result: ^integer_like]

      defop receive_start(), results: [result: ^integer_like]

      defop receive_start_set(value = ^integer_like), results: [result: ^integer_like]

      defop current_entry(), results: [result: ^integer_like]

      defop process_done(result = ^integer_like), results: [result: ^integer_like]

      defop processes_runnable(), results: [result: ^integer_like]

      defop process_result(pid = ^term_like), results: [result: ^integer_like]

      # Parks the current actor when its mailbox has no message beyond the
      # completed selective-receive scan cursor.
      defop process_wait(cursor = ^integer_like), results: [result: ^integer_like]

      # Runs the actor scheduler with a fixed worker count. `dispatcher` is a
      # stable native trampoline with the signature `(pid: i64) -> i64`.
      defop worker_run(worker_count = ^integer_like, dispatcher = ^any_value), results: [result: ^integer_like]

      defop reduction_tick(cost = ^integer_like), results: [result: ^integer_like]

      defop clock_init(budget = ^integer_like), results: [result: ^integer_like]

      defop yield_mark(), results: [result: ^integer_like]
    end
  end
end
