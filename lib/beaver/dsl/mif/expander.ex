defmodule Beaver.MIF.Expander do
  require Logger
  require Beaver.Env
  alias Beaver.MLIR.Attribute
  use Beaver
  alias MLIR.Dialect.{Func, CF}
  require Func
  # Define the environment we will use for expansion.
  # We reset the fields below but we will need to set
  # them accordingly later on.
  @env %{
    Macro.Env.prune_compile_info(__ENV__)
    | line: 0,
      file: "nofile",
      module: nil,
      function: nil,
      context_modules: []
  }
  defp env, do: @env

  # This is a proof of concept of how to build language server
  # tooling or a compiler of a custom language on top of Elixir's
  # building blocks.
  #
  # This example itself will focus on the language server use case.
  # The goal is to traverse expressions collecting information which
  # will be stored in the state (which exists in addition to the
  # environment). The expansion also returns an AST, which has little
  # use for language servers but, in compiler cases, the AST is the
  # one which will be further explored and compiled.
  #
  # For compilers, we'd also need two additional features: the ability
  # to programatically report compiler errors and the ability to
  # track variables. This may be added in the future.
  def expand(ast, file) do
    ctx = MLIR.Context.create()
    available_ops = MapSet.new(MLIR.Dialect.Registry.ops(:all, ctx: ctx))

    mlir = %{
      ctx: ctx,
      mod: nil,
      blk: MLIR.Block.create(),
      available_ops: available_ops,
      vars: Map.new(),
      region: nil
    }

    expand(
      ast,
      %{attrs: [], remotes: [], locals: [], definitions: [], vars: [], mlir: mlir},
      %{env() | file: file}
    )
  end

  def expand_with_mlir(ast, %{ctx: ctx} = mlir, env) do
    available_ops = MapSet.new(MLIR.Dialect.Registry.ops(:all, ctx: ctx))
    mlir = mlir |> Map.put(:available_ops, available_ops)

    expand(
      ast,
      %{attrs: [], remotes: [], locals: [], definitions: [], vars: [], mlir: mlir},
      env
    )
  end

  # The goal of this function is to traverse all of Elixir special
  # forms. The list is actually relatively small and a good reference
  # is the Elixir type checker: https://github.com/elixir-lang/elixir/blob/494a018abbc88901747c32032ec9e2c408f40608/lib/elixir/lib/module/types/expr.ex
  # Everything that is not a special form, is either a local call,
  # a remote call, or a literal.
  #
  # Besides remember that Elixir has two special contexts: match
  # (in pattern matching) and guards. Guards are relatively simple
  # while matches define variables and need to consider both the
  # usage of `^` and `::` specifiers in binaries.

  ## Containers
  # A language server needs to support all types. A custom compiler
  # needs to choose which ones to support. Don't forget that both
  # lists and maps need to consider the usage of `|`. Binaries need
  # to handle `::` (skipped here for convenience).

  defp expand([_ | _] = list, state, env) do
    expand_list(list, state, env)
  end

  defp expand({left, right}, state, env) do
    {left, state, env} = expand(left, state, env)
    {right, state, env} = expand(right, state, env)
    {{left, right}, state, env}
  end

  defp expand({:{}, meta, args}, state, env) do
    {args, state, env} = expand_list(args, state, env)
    {{:{}, meta, args}, state, env}
  end

  defp expand({:%{}, meta, args}, state, env) do
    {args, state, env} = expand_list(args, state, env)
    {{:%{}, meta, args}, state, env}
  end

  defp expand({:|, meta, [left, right]}, state, env) do
    {left, state, env} = expand(left, state, env)
    {right, state, env} = expand(right, state, env)
    {{:|, meta, [left, right]}, state, env}
  end

  defp expand({:<<>>, meta, args}, state, env) do
    {args, state, env} = expand_list(args, state, env)
    {{:<<>>, meta, args}, state, env}
  end

  ## __block__

  defp expand({:__block__, _, list}, state, env) do
    expand_list(list, state, env)
  end

  ## __aliases__

  defp expand({:__aliases__, meta, [head | tail] = list}, state, env) do
    case Macro.Env.expand_alias(env, meta, list, trace: false) do
      {:alias, alias} ->
        # A compiler may want to emit a :local_function trace in here.
        # Elixir also warns on easy to confuse aliases, such as True/False/Nil.
        {alias, state, env}

      :error ->
        with alias <- Module.concat([head]),
             {:ok, found} <- Keyword.fetch(env.aliases, alias) do
          {found, state, env}
        else
          _ ->
            {head, state, env} = expand(head, state, env)

            if is_atom(head) do
              # A compiler may want to emit a :local_function trace in here.
              {Module.concat([head | tail]), state, env}
            else
              {{:__aliases__, meta, [head | tail]}, state, env}
            end
        end
    end
  end

  ## require, alias, import
  # Those are the main special forms and they require some care.
  #
  # First of all, if __aliases__ is changed to emit traces (which a
  # custom compiler should), we should not emit traces when expanding
  # the first argument of require/alias/import.
  #
  # Second, we must never expand the alias in `:as`. This is handled
  # below.
  #
  # Finally, multi-alias/import/require, such as alias Foo.Bar.{Baz, Bat}
  # is not implemented, check elixir_expand.erl on how to implement it.

  defp expand({form, meta, [arg]}, state, env) when form in [:require, :alias, :import] do
    expand({form, meta, [arg, []]}, state, env)
  end

  defp expand({:alias, meta, [arg, opts]}, state, env) do
    {arg, state, env} = expand(arg, state, env)
    {opts, state, env} = expand_directive_opts(opts, state, env)

    case arg do
      {:{}, _, aliases} ->
        for alias <- aliases do
          {:alias, meta, [alias, opts]}
        end
        |> expand(state, env)

      # An actual compiler would raise if the alias fails.
      _ ->
        case Macro.Env.define_alias(env, meta, arg, [trace: false] ++ opts) do
          {:ok, env} -> {arg, state, env}
          {:error, _} -> {arg, state, env}
        end
    end
  end

  defp expand({:require, meta, [arg, opts]}, state, env) do
    {arg, state, env} = expand(arg, state, env)
    {opts, state, env} = expand_directive_opts(opts, state, env)

    # An actual compiler would raise if the module is not defined or if the require fails.
    case Macro.Env.define_require(env, meta, arg, [trace: false] ++ opts) do
      {:ok, env} -> {arg, state, env}
      {:error, _} -> {arg, state, env}
    end
  end

  defp expand({:import, meta, [arg, opts]}, state, env) do
    {arg, state, env} = expand(arg, state, env)
    {opts, state, env} = expand_directive_opts(opts, state, env)

    # An actual compiler would raise if the module is not defined or if the import fails.
    with true <- is_atom(arg) and Code.ensure_loaded?(arg),
         {:ok, env} <- Macro.Env.define_import(env, meta, arg, [trace: false] ++ opts) do
      {arg, state, env}
    else
      _ -> {arg, state, env}
    end
  end

  @intrinsics Beaver.MIF.Prelude.intrinsics()
  defp expand({fun, _meta, [left, right]}, state, env) when fun in @intrinsics do
    {left, state, env} = expand(left, state, env)
    {right, state, env} = expand(right, state, env)

    {Beaver.MIF.Prelude.handle_intrinsic(fun, [left, right],
       ctx: state.mlir.ctx,
       block: state.mlir.blk
     ), state, env}
  end

  ## =/2
  # We include = as an example of how we could handle variables.
  # For example, if you want to store where variables are defined,
  # you would collect this information in expand_pattern/3 and
  # invoke it from all relevant places (such as case, cond, try, etc).

  defp expand({:=, _meta, [left, right]}, state, env) do
    {left, state, env} = expand_pattern(left, state, env)
    {right, state, env} = expand(right, state, env)
    state = put_mlir_var(state, left, right)
    {right, state, env}
  end

  ## quote/1, quote/2
  # We need to expand options and look inside unquote/unquote_splicing.
  # A custom compiler may want to raise on this special form (for example),
  # quoted expressions make no sense if you are writing a language that
  # compiles to C.

  defp expand({:quote, _, [opts]}, state, env) do
    {block, opts} = Keyword.pop(opts, :do)
    {_opts, state, env} = expand_list(opts, state, env)
    expand_quote(block, state, env)
  end

  defp expand({:quote, _, [opts, block_opts]}, state, env) do
    {_opts, state, env} = expand_list(opts, state, env)
    expand_quote(Keyword.get(block_opts, :do), state, env)
  end

  ## Pin operator
  # It only appears inside match and it disables the match behaviour.

  defp expand({:^, _meta, [arg]}, state, %{context: context} = env) do
    {b, state, env} = expand(arg, state, %{env | context: nil})
    match?(%MLIR.Block{}, b) || raise Beaver.EnvNotFoundError, MLIR.Block

    br =
      mlir ctx: state.mlir.ctx, block: state.mlir.blk do
        CF.br({b, []}) >>> []
      end

    {br, state, %{env | context: context}}
  end

  ## Remote call

  defp expand({{:., _dot_meta, [module, fun]}, meta, args}, state, env)
       when is_atom(fun) and is_list(args) do
    {module, state, env} = expand(module, state, env)
    arity = length(args)

    if is_atom(module) do
      case Macro.Env.expand_require(env, meta, module, fun, arity,
             trace: false,
             check_deprecations: false
           ) do
        {:macro, module, callback} ->
          expand_macro(meta, module, fun, args, callback, state, env)

        :error ->
          Code.ensure_loaded(module)

          cond do
            function_exported?(module, :handle_intrinsic, 3) ->
              {args, state, env} = args |> expand(state, env)

              {module.handle_intrinsic(fun, args, ctx: state.mlir.ctx, block: state.mlir.blk),
               state, env}

            module == Beaver.MLIR.Attribute ->
              {args, state, env} = args |> expand(state, env)
              {apply(Beaver.MLIR.Attribute, fun, args), state, env}

            true ->
              raise ArgumentError, "Unknown intrinsic: #{inspect(module)}.#{fun}"
          end
      end
    else
      [{dialect, _, _}, op] = [module, fun]
      op = "#{dialect}.#{op}"

      MapSet.member?(state.mlir.available_ops, op) or
        raise ArgumentError, "Unknown MLIR operation to create: #{op}"

      {args, state, env} = expand_list(args, state, env)

      op =
        %Beaver.SSA{
          op: op,
          arguments: args,
          ctx: state.mlir.ctx,
          block: state.mlir.blk,
          loc: Beaver.MLIR.Location.from_env(env)
        }
        |> MLIR.Operation.create()

      {{MLIR.Operation.results(op), meta, args}, state, env}
    end
  end

  ## Imported or local call

  defp expand({fun, meta, args}, state, env) when is_atom(fun) and is_list(args) do
    arity = length(args)

    # For language servers, we don't want to emit traces, nor expand local macros,
    # nor print deprecation warnings. Compilers likely want those set to true.
    case Macro.Env.expand_import(env, meta, fun, arity,
           trace: true,
           allow_locals: false,
           check_deprecations: true
         ) do
      {:macro, module, callback} ->
        expand_macro(meta, module, fun, args, callback, state, env)

      {:function, module, fun} ->
        expand_remote(meta, module, fun, args, state, env)

      :error ->
        expand_local(meta, fun, args, state, env)
    end
  end

  ## __MODULE__, __DIR__, __ENV__, __CALLER__
  # A custom compiler may want to raise.

  defp expand({:__MODULE__, _, ctx}, state, env) when is_atom(ctx) do
    {env.module, state, env}
  end

  defp expand({:__DIR__, _, ctx}, state, env) when is_atom(ctx) do
    {Path.dirname(env.file), state, env}
  end

  defp expand({:__ENV__, _, ctx}, state, env) when is_atom(ctx) do
    {Macro.escape(env), state, env}
  end

  defp expand({:__CALLER__, _, ctx} = ast, state, env) when is_atom(ctx) do
    {ast, state, env}
  end

  ## var
  # For the language server, we only want to capture definitions,
  # we don't care when they are used.

  defp expand({var, meta, ctx} = ast, state, %{context: :match} = env)
       when is_atom(var) and is_atom(ctx) do
    ctx = Keyword.get(meta, :context, ctx)
    state = update_in(state.vars, &[{var, ctx} | &1])
    {ast, state, env}
  end

  ## Fallback

  defp expand(ast, state, env) do
    {get_mlir_var(state, ast) || ast, state, env}
  end

  defp mangling(mod, func) do
    Module.concat(mod, func)
  end

  ## Macro handling

  # This is going to be the function where you will intercept expansions
  # and attach custom behaviour. As an example, we will capture the module
  # definition, fully replacing the actual implementation. You could also
  # use this to capture module attributes (optionally delegating to the actual
  # implementation), function expansion, and more.
  defp expand_macro(meta, Kernel, :defmodule, [alias, [do: block]], _callback, state, env) do
    {expanded, state, env} = expand(alias, state, env)

    state =
      put_in(
        state.mlir.mod,
        mlir ctx: state.mlir.ctx do
          module sym_name: Attribute.symbol_name(expanded) do
          end
        end
      )

    state = put_in(state.mlir.blk, MLIR.CAPI.mlirModuleGetBody(state.mlir.mod))

    if is_atom(expanded) do
      {full, env} = alias_defmodule(meta, alias, expanded, env)
      env = %{env | context_modules: [full | env.context_modules]}

      # The env inside the block is discarded.
      {result, state, _env} = expand(block, state, %{env | module: full})
      {result, state, env}
    else
      # If we don't know the module name, do we still want to expand it here?
      # Perhaps it would be useful for dealing with local functions anyway?
      # But note that __MODULE__ will return nil.
      #
      # The env inside the block is discarded.
      {result, state, _env} = expand(block, state, env)
      {result, state, env}
    end
  end

  defp expand_macro(_meta, Kernel, :def, [call, [do: body]], _callback, state, env) do
    {:"::", _, [call, ret_types]} = call

    {name, args} = Macro.decompose_call(call)
    name = mangling(env.module, name)

    body =
      body
      |> List.wrap()

    {args, arg_types} =
      for {:"::", _, [a, t]} <- args do
        {a, t}
      end
      |> Enum.unzip()

    f =
      mlir ctx: state.mlir.ctx, block: state.mlir.blk do
        {ret_types, state, env} = ret_types |> expand(state, env)
        {arg_types, state, env} = arg_types |> expand(state, env)
        ft = Type.function(arg_types, ret_types, ctx: Beaver.Env.context())

        Func.func _(sym_name: "\"#{name}\"", function_type: ft) do
          region do
            state = put_in(state.mlir.region, Beaver.Env.region())

            block _entry() do
              MLIR.Block.add_args!(Beaver.Env.block(), arg_types, ctx: Beaver.Env.context())

              arg_values =
                Range.new(0, length(args) - 1)
                |> Enum.map(&MLIR.Block.get_arg!(Beaver.Env.block(), &1))

              state =
                Enum.zip(args, arg_values)
                |> Enum.reduce(state, fn {k, v}, state -> put_mlir_var(state, k, v) end)

              state = put_in(state.mlir.blk, Beaver.Env.block())
              expand(body, state, env)
            end
          end
        end
      end

    {f, state, env}
  end

  defp expand_macro(_meta, Beaver, :block, args, _callback, state, env) do
    b =
      mlir ctx: state.mlir.ctx do
        block do
          [[do: body]] = args
          body |> expand(put_in(state.mlir.blk, Beaver.Env.block()), env)
        end
      end

    MLIR.CAPI.mlirRegionAppendOwnedBlock(state.mlir.region, b)
    {b, state, env}
  end

  defp expand_macro(_meta, Beaver.MIF, :cond_br, [condition, clauses], _callback, state, env) do
    true_body = Keyword.fetch!(clauses, :do)
    false_body = Keyword.fetch!(clauses, :else)
    {condition, state, env} = expand(condition, state, env)

    v =
      mlir ctx: state.mlir.ctx, block: state.mlir.blk do
        true_body =
          block do
            expand(true_body, put_in(state.mlir.blk, Beaver.Env.block()), env)
          end
          |> tap(&MLIR.CAPI.mlirRegionAppendOwnedBlock(state.mlir.region, &1))

        false_body =
          block do
            expand(false_body, put_in(state.mlir.blk, Beaver.Env.block()), env)
          end
          |> tap(&MLIR.CAPI.mlirRegionAppendOwnedBlock(state.mlir.region, &1))

        CF.cond_br(
          condition,
          true_body,
          false_body,
          loc: Beaver.MLIR.Location.from_env(env)
        ) >>> []
      end

    {v, state, env}
  end

  defp expand_macro(_meta, Beaver.MIF, :struct_if, [condition, clauses], _callback, state, env) do
    true_body = Keyword.fetch!(clauses, :do)
    false_body = clauses[:else]
    {condition, state, env} = expand(condition, state, env)

    v =
      mlir ctx: state.mlir.ctx, block: state.mlir.blk do
        alias Beaver.MLIR.Dialect.SCF

        SCF.if [condition] do
          region do
            block _true() do
              expand(true_body, put_in(state.mlir.blk, Beaver.Env.block()), env)
              SCF.yield() >>> []
            end
          end

          region do
            block _false() do
              if false_body do
                expand(false_body, put_in(state.mlir.blk, Beaver.Env.block()), env)
              end

              SCF.yield() >>> []
            end
          end
        end >>> []
      end

    {v, state, env}
  end

  defp expand_macro(_meta, Beaver.MIF, :while_loop, [expr, [do: body]], _callback, state, env) do
    v =
      mlir ctx: state.mlir.ctx, block: state.mlir.blk do
        Beaver.MLIR.Dialect.SCF.while [] do
          region do
            block _() do
              {condition, _state, _env} =
                expand(expr, put_in(state.mlir.blk, Beaver.Env.block()), env)

              Beaver.MLIR.Dialect.SCF.condition(condition) >>> []
            end
          end

          region do
            block _() do
              expand(body, put_in(state.mlir.blk, Beaver.Env.block()), env)
              Beaver.MLIR.Dialect.SCF.yield() >>> []
            end
          end
        end >>> []
      end

    {v, state, env}
  end

  defp expand_macro(_meta, Beaver.MIF, :for_loop, [expr, [do: body]], _callback, state, env) do
    {:<-, _, [{element, index}, {:{}, _, [t, ptr, len]}]} = expr
    {len, state, env} = expand(len, state, env)
    {t, state, env} = expand(t, state, env)
    {ptr, state, env} = expand(ptr, state, env)

    v =
      mlir ctx: state.mlir.ctx, block: state.mlir.blk do
        alias Beaver.MLIR.Dialect.{Index, SCF, LLVM}
        zero = Index.constant(value: Attribute.index(0)) >>> Type.index()
        lower_bound = zero
        upper_bound = Index.casts(len) >>> Type.index()
        step = Index.constant(value: Attribute.index(1)) >>> Type.index()

        SCF.for [lower_bound, upper_bound, step] do
          region do
            block _body(index_val >>> Type.index()) do
              index_casted = Index.casts(index_val) >>> Type.i64()

              element_ptr =
                LLVM.getelementptr(ptr, index_casted,
                  elem_type: t,
                  rawConstantIndices: ~a{array<i32: -2147483648>}
                ) >>> ~t{!llvm.ptr}

              element_val = LLVM.load(element_ptr) >>> t
              state = put_mlir_var(state, element, element_val)
              state = put_mlir_var(state, index, index_val)
              expand(body, put_in(state.mlir.blk, Beaver.Env.block()), env)
              Beaver.MLIR.Dialect.SCF.yield() >>> []
            end
          end
        end >>> []
      end

    {v, state, env}
  end

  defp expand_macro(_meta, Beaver.MIF, :op, [call], _callback, state, env) do
    {call, return_types} = Beaver.MIF.decompose_call_and_returns(call)
    {{dialect, _, _}, op, args} = Macro.decompose_call(call)
    op = "#{dialect}.#{op}"
    {args, state, env} = expand(args, state, env)
    {[return_types], state, env} = expand(return_types, state, env)

    op =
      %Beaver.SSA{
        op: op,
        arguments: args,
        ctx: state.mlir.ctx,
        block: state.mlir.blk,
        loc: Beaver.MLIR.Location.from_env(env)
      }
      |> Beaver.SSA.put_results(return_types)
      |> MLIR.Operation.create()

    {op, state, env}
  end

  defp expand_macro(meta, Beaver.MIF, :value, [call], callback, state, env) do
    {op, state, env} =
      expand_macro(meta, Beaver.MIF, :op, [call], callback, state, env)

    {MLIR.Operation.results(op), state, env}
  end

  defp expand_macro(
         meta,
         Beaver.MIF,
         :call,
         [{:"::", type_meta, [call, types]}],
         callback,
         state,
         env
       ) do
    expand_macro(
      meta,
      Beaver.MIF,
      :call,
      [env.module, {:"::", type_meta, [call, types]}],
      callback,
      state,
      env
    )
  end

  defp expand_macro(
         _meta,
         Beaver.MIF,
         :call,
         [mod, {:"::", _, [call, types]}],
         _callback,
         state,
         env
       ) do
    {mod, state, env} = expand(mod, state, env)
    {name, args} = Macro.decompose_call(call)
    name = mangling(mod, name)
    {args, state, env} = expand(args, state, env)

    # TODO: flatten should be unnecessary, refactor to support call Mod.fun(args) instead of call Mod, fun(args)
    args = args |> List.flatten()
    {types, state, env} = expand(types, state, env)

    op =
      mlir ctx: state.mlir.ctx, block: state.mlir.blk do
        %Beaver.SSA{
          op: "func.call",
          arguments: args ++ [callee: Attribute.flat_symbol_ref("#{name}")],
          ctx: Beaver.Env.context(),
          block: Beaver.Env.block(),
          loc: Beaver.MLIR.Location.from_env(env)
        }
        |> Beaver.SSA.put_results(types)
        |> MLIR.Operation.create()
      end

    {MLIR.Operation.results(op), state, env}
  end

  defp expand_macro(
         meta,
         Beaver.MIF,
         :call,
         [call],
         callback,
         state,
         env
       ) do
    expand_macro(
      meta,
      Beaver.MIF,
      :call,
      [env.module, {:"::", meta, [call, []]}],
      callback,
      state,
      env
    )
  end

  defp expand_macro(meta, module, fun, args, callback, state, env) do
    case {module, fun} do
      mf when mf in [{Beaver, :block}, {Beaver, :mlir}] ->
        expand_intrinsic_macro_callback(meta, module, fun, args, callback, state, env)

      {Beaver.MIF, _} ->
        expand_intrinsic_macro_callback(meta, module, fun, args, callback, state, env)

      _ ->
        expand_macro_callback(meta, module, fun, args, callback, state, env)
    end
  end

  defp expand_intrinsic_macro_callback(_meta, module, fun, args, _callback, _state, _env) do
    raise ArgumentError, "TODO: #{inspect(module)}.#{fun}, #{inspect(args, pretty: true)}"
    # callback.(meta, args) |> expand(state, env)
  end

  defp expand_macro_callback(meta, _module, _fun, args, callback, state, env) do
    callback.(meta, args) |> expand(state, env)
  end

  ## defmodule helpers
  # defmodule automatically defines aliases, we need to mirror this feature here.

  # defmodule Elixir.Alias
  defp alias_defmodule(_meta, {:__aliases__, _, [:"Elixir", _ | _]}, module, env),
    do: {module, env}

  # defmodule Alias in root
  defp alias_defmodule(_meta, {:__aliases__, _, _}, module, %{module: nil} = env),
    do: {module, env}

  # defmodule Alias nested
  defp alias_defmodule(meta, {:__aliases__, _, [h | t]}, _module, env) when is_atom(h) do
    module = Module.concat([env.module, h])
    alias = String.to_atom("Elixir." <> Atom.to_string(h))
    {:ok, env} = Macro.Env.define_alias(env, meta, module, as: alias, trace: false)

    case t do
      [] -> {module, env}
      _ -> {String.to_atom(Enum.join([module | t], ".")), env}
    end
  end

  # defmodule _
  defp alias_defmodule(_meta, _raw, module, env) do
    {module, env}
  end

  ## Helpers

  @intrinsics Beaver.MIF.Prelude.intrinsics()
  defp expand_remote(_meta, Kernel, fun, args, state, env) when fun in @intrinsics do
    {args, state, env} = expand(args, state, env)

    {Beaver.MIF.Prelude.handle_intrinsic(fun, args,
       ctx: state.mlir.ctx,
       block: state.mlir.blk
     ), state, env}
  end

  defp expand_remote(meta, module, fun, args, state, env) do
    # A compiler may want to emit a :remote_function trace in here.
    state = update_in(state.remotes, &[{module, fun, length(args)} | &1])
    {args, state, env} = expand_list(args, state, env)

    if module in [MLIR.Type] do
      {apply(module, fun, [[ctx: state.mlir.ctx]]), state, env}
    else
      {{{:., meta, [module, fun]}, meta, args}, state, env}
    end
  end

  defp expand_local(meta, fun, args, state, env) do
    # A compiler may want to emit a :local_function trace in here.
    state = update_in(state.locals, &[{fun, length(args)} | &1])
    {args, state, env} = expand_list(args, state, env)
    Code.ensure_loaded!(MLIR.Type)

    if function_exported?(MLIR.Type, fun, 1) do
      {apply(MLIR.Type, fun, [[ctx: state.mlir.ctx]]), state, env}
    else
      case i =
             Beaver.MIF.Prelude.handle_intrinsic(fun, args,
               ctx: state.mlir.ctx,
               block: state.mlir.blk
             ) do
        :not_handled ->
          expand_macro(
            meta,
            Beaver.MIF,
            :call,
            [{:"::", meta, [quote(do: unquote(fun)(unquote(args))), []]}],
            nil,
            state,
            env
          )

        _ ->
          {i, state, env}
      end
    end
  end

  defp expand_pattern(pattern, state, %{context: context} = env) do
    {pattern, state, env} = expand(pattern, state, %{env | context: :match})
    {pattern, state, %{env | context: context}}
  end

  defp expand_directive_opts(opts, state, env) do
    opts =
      Keyword.replace_lazy(opts, :as, fn
        {:__aliases__, _, list} -> Module.concat(list)
        other -> other
      end)

    expand(opts, state, env)
  end

  defp expand_list(ast, state, env), do: expand_list(ast, state, env, [])

  defp expand_list([], state, env, acc) do
    {Enum.reverse(acc), state, env}
  end

  defp expand_list([h | t], state, env, acc) do
    {h, state, env} = expand(h, state, env)
    expand_list(t, state, env, [h | acc])
  end

  defp expand_quote(ast, state, env) do
    {_, {state, env}} =
      Macro.prewalk(ast, {state, env}, fn
        # We need to traverse inside unquotes
        {unquote, _, [expr]}, {state, env} when unquote in [:unquote, :unquote_splicing] ->
          {_expr, state, env} = expand(expr, state, env)
          {:ok, {state, env}}

        # If we find a quote inside a quote, we stop traversing it
        {:quote, _, [_]}, acc ->
          {:ok, acc}

        {:quote, _, [_, _]}, acc ->
          {:ok, acc}

        # Otherwise we go on
        node, acc ->
          {node, acc}
      end)

    {ast, state, env}
  end

  defp put_mlir_var(state, {name, _meta, _ctx} = _ast, val) do
    update_in(state.mlir.vars, &Map.put(&1, name, val))
  end

  defp get_mlir_var(state, {name, _meta, _ctx} = _ast) do
    Map.get(state.mlir.vars, name)
  end

  defp get_mlir_var(_state, _ast) do
    nil
  end
end
