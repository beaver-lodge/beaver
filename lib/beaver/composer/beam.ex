defmodule Beaver.Composer.Beam do
  @moduledoc false
  alias Beaver.{Composer, MLIR, Walker}
  alias Beaver.MLIR.Operation

  # State is threaded through the execution tree, including failure paths.
  # Neither an Agent nor a native callback is needed for a BEAM pass.
  def run(%Composer{op: ir, passes: passes}, opts) do
    unless Keyword.get(opts, :verifier, true) in [true, false, :pipeline],
      do: raise(ArgumentError, ":verifier must be true, false, or :pipeline")

    op = Operation.from_module(ir)
    ctx = MLIR.context(op)

    case prepare(passes, ctx, Operation.name(op), false, opts) do
      {:ok, stages} ->
        {status, stages, diagnostics} = execute(stages, op, opts)

        {status, diagnostics} =
          if status == :ok and opts[:verifier] == :pipeline do
            case verify(op, verifier: true) do
              :ok -> {:ok, diagnostics}
              {:error, errors} -> {:error, diagnostics ++ errors}
            end
          else
            {status, diagnostics}
          end

        cleanup = destroy(stages)

        if status == :ok and cleanup == [],
          do: {:ok, ir, diagnostics},
          else: {:error, diagnostics ++ cleanup}

      {:error, diagnostics} ->
        {:error, diagnostics}
    end
  end

  defp prepare(passes, ctx, anchor, nested?, opts) do
    passes
    |> Enum.chunk_by(&native?/1)
    |> Enum.flat_map(fn group -> if native?(hd(group)), do: [{:native, group}], else: group end)
    |> Enum.reduce_while({:ok, []}, fn pass, {:ok, acc} ->
      case prepare_stage(pass, ctx, anchor, nested?, opts) do
        {:ok, stage} -> {:cont, {:ok, [stage | acc]}}
        {:error, diagnostics} -> {:halt, {:error, diagnostics ++ destroy(acc)}}
      end
    end)
    |> case do
      {:ok, stages} -> {:ok, Enum.reverse(stages)}
      error -> error
    end
  end

  defp native?(pass), do: is_binary(pass) or is_struct(pass)

  defp prepare_stage({:native, passes}, ctx, anchor, _nested?, opts) do
    opts = Keyword.put(opts, :verifier, Keyword.get(opts, :verifier, true) == true)

    case capture(fn -> Composer.init(%Composer{ctx: ctx, passes: passes}, opts, anchor) end) do
      {:ok, pm} ->
        if opts[:print], do: MLIR.PassManager.enable_ir_printing(pm)
        {:ok, {:native, pm}}

      error ->
        error
    end
  end

  defp prepare_stage({root, passes}, ctx, _anchor, _nested?, opts) when is_binary(root) and is_list(passes) do
    if MLIR.Trait.has?(ctx, root, :isolated_from_above) do
      case prepare(passes, ctx, root, true, opts) do
        {:ok, stages} -> {:ok, {:nested, root, stages}}
        error -> error
      end
    else
      failure("nested BEAM pipeline requires a registered IsolatedFromAbove operation: #{root}")
    end
  end

  defp prepare_stage(pass, ctx, anchor, nested?, _opts) do
    with {:ok, callbacks} <- capture(fn -> callbacks(pass, anchor) end),
         {:ok, constructed} <- capture(fn -> callbacks.construct.(nil) end) do
      case capture(fn -> callbacks.initialize.(ctx, constructed) end) do
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
        case capture(fn -> callbacks.construct.(clone) end) do
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

    %{
      construct: &Beaver.FallbackPass.construct/1,
      initialize: &Beaver.FallbackPass.initialize/2,
      clone: &Beaver.FallbackPass.clone/1,
      destruct: &Beaver.FallbackPass.destruct/1,
      run: run
    }
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
    case capture(fn -> callbacks.run.(op, value) end) do
      {:ok, value} ->
        stage = {:beam, callbacks, value, originals}

        case verify(op, opts) do
          :ok -> {:ok, stage, []}
          {:error, diagnostics} -> {:error, stage, diagnostics}
        end

      {:error, diagnostics} ->
        {:error, stage, diagnostics}
    end
  end

  defp execute_stage({:native, pm} = stage, op, _opts) do
    case capture(fn -> MLIR.PassManager.run(pm, op) end) do
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

    {status, {:nested, root, stages}, diagnostics}
  end

  defp verify(op, opts) do
    if Keyword.get(opts, :verifier, true) == true do
      case capture(fn -> MLIR.verify(op) end) do
        {:ok, {:ok, _}} -> :ok
        {:ok, {:error, diagnostics}} -> {:error, diagnostics}
        {:error, diagnostics} -> {:error, diagnostics}
      end
    else
      :ok
    end
  end

  defp destroy(stages) do
    Enum.flat_map(stages, fn
      {:native, pm} ->
        cleanup(fn -> MLIR.PassManager.destroy(pm) end)

      {:nested, _, nested} ->
        destroy(nested)

      {:beam, callbacks, value, originals} ->
        Enum.flat_map([value | originals], &destroy_value(callbacks, &1))
    end)
  end

  defp destroy_value(callbacks, value), do: cleanup(fn -> callbacks.destruct.(value) end)

  defp cleanup(fun) do
    case capture(fun) do
      {:ok, _} -> []
      {:error, diagnostics} -> diagnostics
    end
  end

  defp capture(fun) do
    {:ok, fun.()}
  catch
    kind, reason -> failure(Exception.format(kind, reason, __STACKTRACE__))
  end

  defp failure(message), do: {:error, diagnostics(message)}
  defp diagnostics(message), do: [{:error, "loc(unknown)", message, []}]
end
