defmodule ENIFSendMsg do
  @moduledoc false
  alias Beaver.ENIF
  use Beaver
  alias MLIR.Dialect.{Func, Ptr, MemRef}
  require Func
  use ENIFSupport

  @impl ENIFSupport
  def create(ctx) do
    mlir ctx: ctx do
      module do
        env_t = ENIF.Type.env(ctx: ctx)
        term_t = ENIF.Type.term()
        pid_t = ENIF.Type.pid(ctx: ctx)
        ptr_t = Ptr.type()
        generic_space = Ptr.generic_space()

        Func.func send(function_type: Type.function([env_t, term_t, term_t], [term_t])) do
          region do
            block _(env >>> env_t, pid_term >>> term_t, msg >>> term_t) do
              pid_ptr =
                MemRef.alloca(operand_segment_sizes: :infer) >>>
                  Type.memref!([1], pid_t, memory_space: generic_space)

              pid = Ptr.to_ptr(pid_ptr) >>> ptr_t
              null_env = Ptr.constant(value: Ptr.null(type: ptr_t)) >>> ptr_t

              ENIF.get_local_pid(env, pid_term, pid) >>> :infer
              ENIF.send(env, pid, null_env, msg) >>> :infer
              Func.return(msg) >>> []
            end
          end
        end
      end
    end
  end
end
