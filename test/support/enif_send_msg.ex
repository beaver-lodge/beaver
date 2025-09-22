defmodule ENIFSendMsg do
  @moduledoc false
  alias Beaver.ENIF
  use Beaver
  alias MLIR.Dialect.{Func, LLVM, Arith}
  require Func
  use ENIFSupport

  @impl ENIFSupport
  def create(ctx) do
    mlir ctx: ctx do
      module do
        Beaver.ENIF.declare_external_functions(Beaver.Env.context(), Beaver.Env.block())
        env_t = ENIF.Type.env()
        term_t = ENIF.Type.term()
        pid_t = ENIF.Type.pid()

        Func.func send(function_type: Type.function([env_t, term_t, term_t], [term_t])) do
          region do
            block _(env >>> env_t, pid_term >>> term_t, msg >>> term_t) do
              one = Arith.constant(value: Attribute.integer(Type.i32(), 1)) >>> :infer
              pid = LLVM.alloca(one, elem_type: pid_t) >>> Type.llvm_pointer()
              null_env = ENIF.alloc_env() >>> :infer
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
