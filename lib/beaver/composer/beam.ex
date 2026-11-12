defmodule Beaver.Composer.Beam do
  @moduledoc false
  alias Beaver.{Composer, MLIR, Walker}
  alias Beaver.MLIR.Operation

  alias Beaver.Execution.Profile

  def prepare(%Composer{op: ir, ctx: ctx, passes: passes}, opts) do
    op = if ir, do: Operation.from_module(ir)
    ctx = ctx || MLIR.context(op)
    anchor = if op, do: Operation.name(op), else: Keyword.get(opts, :root, "builtin.module")
    %Composer.Prepared{context: ctx, root: anchor, stages: templates(passes, ctx, anchor, false)}
  end

  def run(composer, opts) do
    case measure(opts, :prepare, fn -> prepare(composer, opts) end) do
      {:ok, plan} -> run_prepared(plan, composer.op, opts)
      error -> error
    end
  end

  def run_prepared(%Composer.Prepared{} = plan, ir, opts) do
    unless Keyword.get(opts, :verifier, true) in [true, false, :pipeline],
      do: raise(ArgumentError, ":verifier must be true, false, or :pipeline")

    op = Operation.from_module(ir)
    MLIR.Context.ensure_same!(op, plan.context)
    if Operation.name(op) != plan.root, do: raise(ArgumentError, "prepared pipeline root mismatch")

    case Profile.measure(opts[:profile], :initialize, fn -> initialize(plan.stages, plan.context, opts) end) do
      {:ok, stages} ->
        {status, stages, diagnostics} = execute(stages, op, opts)
        pipeline_opts = Keyword.put(opts, :verifier, opts[:verifier] == :pipeline)
        {status, diagnostics} = verified({status, diagnostics}, op, pipeline_opts)
        cleanup = Profile.measure(opts[:profile], :cleanup, fn -> destroy(stages) end)
        if status == :ok and cleanup == [], do: {:ok, ir, diagnostics}, else: {:error, diagnostics ++ cleanup}

      {:error, diagnostics} ->
        {:error, diagnostics}
    end
  end

  defp templates(passes, ctx, anchor, nested?) do
    passes
    |> Enum.chunk_by(&native?/1)
    |> Enum.flat_map(fn group -> if native?(hd(group)), do: [{:native, group, anchor}], else: group end)
    |> Enum.map(fn
      {:native, _, _} = native ->
        native

      {root, passes} when is_binary(root) and is_list(passes) ->
        unless MLIR.Trait.has?(ctx, root, :isolated_from_above),
          do: raise(ArgumentError, "nested BEAM pipeline requires a registered IsolatedFromAbove operation: #{root}")

        {:nested, root, templates(passes, ctx, root, true)}

      pass ->
        {:beam, callbacks(pass, anchor), nested?}
    end)
  end

  defp native?(pass), do: is_binary(pass) or is_struct(pass)

  defp initialize(templates, ctx, opts) do
    Enum.reduce_while(templates, {:ok, []}, fn template, {:ok, acc} ->
      case initialize_stage(template, ctx, opts) do
        {:ok, stage} -> {:cont, {:ok, [stage | acc]}}
        {:error, diagnostics} -> {:halt, {:error, diagnostics ++ destroy(acc)}}
      end
    end)
    |> case do
      {:ok, stages} -> {:ok, Enum.reverse(stages)}
      error -> error
    end
  end

  defp initialize_stage({:native, passes, anchor}, ctx, opts) do
    opts = Keyword.put(opts, :verifier, Keyword.get(opts, :verifier, true) == true)

    with {:ok, pm} <- capture(fn -> Composer.init(%Composer{ctx: ctx, passes: passes}, opts, anchor) end) do
      if opts[:print], do: MLIR.PassManager.enable_ir_printing(pm)
      {:ok, {:native, pm}}
    end
  end

  defp initialize_stage({:nested, root, templates}, ctx, opts) do
    with {:ok, stages} <- initialize(templates, ctx, opts), do: {:ok, {:nested, root, stages}}
  end

  defp initialize_stage({:beam, callbacks, nested?}, ctx, _opts) do
    with {:ok, constructed} <- capture(fn -> callbacks.construct.(nil) end) do
      initialized = capture(fn -> callbacks.initialize.(ctx, constructed) end)

      case initialized do
        {:ok, {:ok, value}} ->
          prepare_clone(callbacks, value, nested?)

        {:ok, {:error, value}} ->
          {:error,
           diagnostics("Pass initialization callback returned {:error, state}") ++ destroy_value(callbacks, value)}

        {:ok, other} ->
          {:error,
           diagnostics("invalid pass initialization result: #{inspect(other)}") ++ destroy_value(callbacks, constructed)}

        {:error, errors} ->
          {:error, errors ++ destroy_value(callbacks, constructed)}
      end
    end
  end

  defp prepare_clone(callbacks, value, false), do: {:ok, {:beam, callbacks, value, []}}

  defp prepare_clone(callbacks, value, true) do
    case capture(fn -> callbacks.clone.(value) end) do
      {:ok, clone} ->
        constructed = capture(fn -> callbacks.construct.(clone) end)

        case constructed do
          {:ok, constructed} -> {:ok, {:beam, callbacks, constructed, [value]}}
          {:error, errors} -> {:error, errors ++ destroy_value(callbacks, clone) ++ destroy_value(callbacks, value)}
        end

      {:error, errors} ->
        {:error, errors ++ destroy_value(callbacks, value)}
    end
  end

  defp callbacks({name, root, run}, anchor) when is_function(run) do
    if root != anchor, do: raise(ArgumentError, "pass #{name} targets #{root}, pipeline targets #{anchor}")
    run = if is_function(run, 1), do: fn op, _ -> run.(op) end, else: run
    unless is_function(run, 2), do: raise(ArgumentError, "pass must accept one or two arguments")
    MLIR.Pass.callbacks(run: run)
  end

  defp callbacks(module, anchor) when is_atom(module) do
    Code.ensure_loaded!(module)
    root = module.__info__(:attributes) |> Keyword.get(:root_op, ["builtin.module"]) |> hd()
    if root != anchor, do: raise(ArgumentError, "pass #{inspect(module)} targets #{root}, pipeline targets #{anchor}")

    Map.new([:construct, :initialize, :clone, :destruct, :run], fn name ->
      arity = if name in [:initialize, :run], do: 2, else: 1
      provider = if function_exported?(module, name, arity), do: module, else: Beaver.FallbackPass
      {name, Function.capture(provider, name, arity)}
    end)
  end

  defp execute([], _op, _opts), do: {:ok, [], []}

  defp execute([stage | rest], op, opts) do
    case execute_stage(stage, op, opts) do
      {:ok, stage, diagnostics} ->
        {status, rest, following} = execute(rest, op, opts)
        {status, [stage | rest], diagnostics ++ following}

      {:error, stage, diagnostics} ->
        {:error, [stage | rest], diagnostics}
    end
  end

  defp execute_stage({:beam, callbacks, value, originals} = stage, op, opts) do
    case measure(opts, :beam_pass, fn -> callbacks.run.(op, value) end) do
      {:ok, value} ->
        stage = {:beam, callbacks, value, originals}
        {status, diagnostics} = verified({:ok, []}, op, opts)
        {status, stage, diagnostics}

      {:error, diagnostics} ->
        {:error, stage, diagnostics}
    end
  end

  defp execute_stage({:native, pm} = stage, op, opts) do
    case capture(fn -> MLIR.PassManager.run(pm, op, Keyword.take(opts, [:profile])) end) do
      {:ok, {status, diagnostics}} -> {status, stage, diagnostics}
      {:error, diagnostics} -> {:error, stage, diagnostics}
    end
  end

  defp execute_stage({:nested, root, stages}, op, opts) do
    children =
      for region <- Walker.regions(op),
          block <- Walker.blocks(region),
          child <- Walker.operations(block),
          Operation.name(child) == root,
          do: child

    {status, stages, diagnostics} =
      Enum.reduce_while(children, {:ok, stages, []}, fn child, {:ok, stages, diagnostics} ->
        {status, stages, current} = execute(stages, child, opts)
        result = {status, stages, diagnostics ++ current}
        if status == :ok, do: {:cont, result}, else: {:halt, result}
      end)

    # Mirror MLIR's adaptor pass: children were already checked, but parent
    # invariants (for example symbol uniqueness) still need verification.
    {status, diagnostics} = verified({status, diagnostics}, op, Keyword.put(opts, :recursive, false))
    {status, {:nested, root, stages}, diagnostics}
  end

  defp verified({:error, _} = result, _op, _opts), do: result

  defp verified({:ok, diagnostics}, op, opts) do
    if Keyword.get(opts, :verifier, true) == true do
      verify = fn ->
        MLIR.verify(op,
          strategy: Keyword.get(opts, :verify_execution, :auto),
          recursive: Keyword.get(opts, :recursive, true),
          profile: opts[:profile]
        )
      end

      case capture(verify) do
        {:ok, {:ok, _}} -> {:ok, diagnostics}
        {:ok, {:error, errors}} -> {:error, diagnostics ++ errors}
        {:error, errors} -> {:error, diagnostics ++ errors}
      end
    else
      {:ok, diagnostics}
    end
  end

  defp destroy(stages) do
    Enum.flat_map(stages, fn
      {:native, pm} -> cleanup(fn -> MLIR.PassManager.destroy(pm) end)
      {:nested, _, nested} -> destroy(nested)
      {:beam, callbacks, value, originals} -> Enum.flat_map([value | originals], &destroy_value(callbacks, &1))
    end)
  end

  defp destroy_value(callbacks, value), do: cleanup(fn -> callbacks.destruct.(value) end)

  defp cleanup(fun) do
    case capture(fun) do
      {:ok, _} -> []
      {:error, diagnostics} -> diagnostics
    end
  end

  defp measure(opts, phase, fun), do: capture(fn -> Profile.measure(opts[:profile], phase, fun) end)

  defp capture(fun) do
    {:ok, fun.()}
  catch
    kind, reason -> failure(Exception.format(kind, reason, __STACKTRACE__))
  end

  defp failure(message), do: {:error, diagnostics(message)}
  defp diagnostics(message), do: [{:error, "loc(unknown)", message, []}]
end
