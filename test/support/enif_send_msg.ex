defmodule ENIFSendMsg do
  @moduledoc false
  alias Beaver.ENIF
  use Beaver
  alias MLIR.Dialect.{Func, Ptr, MemRef, LLVM, Builtin}
  require Func
  use ENIFSupport

  @impl ENIFSupport
  def create(ctx) do
    mlir ctx: ctx do
      module do
        Beaver.ENIF.declare_external_functions(Beaver.Env.context(), Beaver.Env.block())
        env_t = ENIF.Type.env(ctx: ctx)
        term_t = ENIF.Type.term()
        pid_t = ENIF.Type.pid(ctx: ctx)

        Func.func send(function_type: Type.function([env_t, term_t, term_t], [term_t])) do
          region do
            block _(env >>> env_t, pid_term >>> term_t, msg >>> term_t) do
              pid_ptr =
                MemRef.alloca(operand_segment_sizes: :infer) >>>
                  Type.memref!([1], pid_t, memory_space: ~a{#ptr.generic_space})

              pid = Ptr.to_ptr(pid_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}
              null_env = LLVM.mlir_zero() >>> Type.llvm_pointer()

              null_env =
                Builtin.unrealized_conversion_cast(null_env) >>> ~t{!ptr.ptr<#ptr.generic_space>}

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
