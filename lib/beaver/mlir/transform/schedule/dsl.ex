alias Beaver.MLIR.{Attribute, Block, CAPI, Operation, Type, Value}
alias Beaver.Deferred

defmodule Beaver.MLIR.Transform.Schedule.DSL do
  @moduledoc """
  An Elixir authoring frontend for upstream Transform dialect schedules.

  The DSL creates a verified `Beaver.MLIR.Module` in a caller-owned context.
  Its helpers create ordinary Transform dialect operations immediately; there
  is no second schedule graph to synchronize with the IR.

  A schedule module can mix the focused helpers in this module with Beaver's
  normal SSA syntax:

      defmodule MySchedule do
        use Beaver.MLIR.Transform.Schedule.DSL

        defschedule search do
          sequence "__transform_main", [root >>> any_op()] do
            _tile = knob("tile", [8, 16], type: param(MLIR.Type.i64()))
            # Any existing Beaver SSA operation may be emitted here.
            yield()
          end
        end
      end

      schedule = MySchedule.search(ctx: context)

  The caller owns both `context` and the returned module.
  """

  alias __MODULE__, as: DSL
  alias Beaver.MLIR
  alias Beaver.MLIR.Transform

  @doc "Imports `defschedule` and the schedule authoring helpers."
  defmacro __using__(_opts) do
    quote do
      use Beaver
      require Beaver.Env
      import DSL
    end
  end

  @doc """
  Defines a zero-arity schedule builder with an optional keyword argument.

  The generated function requires `:ctx` to be a caller-owned
  `Beaver.MLIR.Context`. It returns a verified `Beaver.MLIR.Module`; the caller
  must destroy the module before destroying the context.
  """
  defmacro defschedule(call, do: body) do
    name = schedule_function_name!(call)
    caller = Macro.escape(__CALLER__)

    quote do
      def unquote(name)(opts \\ []) do
        DSL.__build_schedule__(
          opts,
          unquote(caller),
          fn context, block ->
            # Establish only the outer module insertion environment here. The
            # sequence helper runs Beaver's SSA prewalk for its own body; doing
            # so at both levels would transform nested block arguments twice.
            Kernel.var!(beaver_internal_env_ctx) = context
            Kernel.var!(beaver_internal_env_ip) = block
            unquote(body)
            _ = Kernel.var!(beaver_internal_env_ctx)
            _ = Kernel.var!(beaver_internal_env_ip)
          end
        )
      end
    end
  end

  defp schedule_function_name!({name, _, args}) when is_atom(name) and args in [[], nil], do: name
  defp schedule_function_name!(name) when is_atom(name), do: name

  defp schedule_function_name!(other),
    do: raise(ArgumentError, "defschedule expects a zero-arity function name, got: #{Macro.to_string(other)}")

  @doc false
  def __build_schedule__(opts, caller, build_body) when is_function(build_body, 2) do
    context = schedule_context!(opts)
    location = source_location(caller, context)
    module = MLIR.Module.empty(location)

    try do
      module
      |> Operation.from_module()
      |> put_unit_attribute("transform.with_named_sequence", context)

      build_body.(context, MLIR.Module.body(module))
      ensure_named_sequence!(module)
      verify_schedule!(module)
    rescue
      exception ->
        MLIR.Module.destroy(module)
        reraise exception, __STACKTRACE__
    catch
      kind, reason ->
        MLIR.Module.destroy(module)
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  defp schedule_context!(opts) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "schedule options must be a keyword list, got: #{inspect(opts)}")

    case Keyword.keys(opts) -- [:ctx] do
      [] -> :ok
      unsupported -> raise ArgumentError, "unsupported schedule options: #{inspect(unsupported)}"
    end

    case Keyword.get(opts, :ctx) do
      %MLIR.Context{} = context ->
        context

      value ->
        raise ArgumentError,
              "defschedule requires a caller-owned %Beaver.MLIR.Context{} via ctx: context, " <>
                "got: #{inspect(value)}"
    end
  end

  defp put_unit_attribute(op, name, ctx) do
    Operation.put_attribute(op, name, Attribute.unit(ctx: ctx))

    op
  end

  defp ensure_named_sequence!(module) do
    found? =
      module
      |> MLIR.Module.body()
      |> Beaver.Walker.operations()
      |> Enum.any?(&(Operation.name(&1) == "transform.named_sequence"))

    unless found?, do: raise(ArgumentError, "defschedule must declare at least one sequence")
  end

  defp verify_schedule!(module) do
    case MLIR.verify(module) do
      {:ok, verified} ->
        verified

      :null ->
        raise ArgumentError, "transform schedule verification failed: module is null"

      {:error, diagnostics} ->
        raise ArgumentError,
              MLIR.Diagnostic.format(diagnostics, "transform schedule verification failed")
    end
  end

  @doc "Declares the default `__transform_main` sequence with one `any_op` root handle."
  defmacro sequence(do: body),
    do: build_sequence_ast("__transform_main", default_root_argument(), body, __CALLER__)

  @doc "Declares a named sequence with one default `any_op` root handle."
  defmacro sequence(name, do: body),
    do: build_sequence_ast(literal_name!(name, "sequence"), default_root_argument(), body, __CALLER__)

  @doc """
  Declares a named sequence with typed handle arguments.

  Arguments use Beaver's `>>>` spelling, for example
  `[root >>> any_op(), function >>> operation("func.func")]`. Sequence results
  are intentionally expressed by ordinary Transform SSA operations and
  `yield/1`; this helper currently declares result-free entry points.
  """
  defmacro sequence(name, arguments, do: body),
    do: build_sequence_ast(literal_name!(name, "sequence"), arguments, body, __CALLER__)

  defp helper_ast(function, arguments, caller) do
    quote do
      DSL.unquote(function)(
        Beaver.Env.context(),
        Beaver.Env.ip(),
        unquote_splicing(arguments),
        unquote(Macro.escape(caller))
      )
    end
  end

  defp default_root_argument do
    root = Macro.var(:root, nil)

    quote do
      [unquote(root) >>> DSL.any_op()]
    end
  end

  defp build_sequence_ast(name, arguments_ast, body, caller) do
    {type_builders, bindings} = sequence_arguments!(arguments_ast)
    caller = Macro.escape(caller)

    quote do
      DSL.__build_sequence__(
        Beaver.Env.context(),
        Beaver.Env.ip(),
        unquote(name),
        unquote(type_builders),
        unquote(caller),
        fn context, sequence_block ->
          Beaver.mlir ctx: context, ip: sequence_block do
            unquote_splicing(bindings)
            unquote(body)
          end
        end
      )
    end
  end

  defp sequence_arguments!(arguments) when is_list(arguments) and arguments != [] do
    arguments
    |> Enum.with_index()
    |> Enum.map(fn {argument, index} -> sequence_argument!(argument, index) end)
    |> Enum.unzip()
  end

  defp sequence_arguments!([]), do: {[], []}

  defp sequence_arguments!(other),
    do: raise(ArgumentError, "sequence arguments must be a literal list, got: #{Macro.to_string(other)}")

  # The surrounding `Beaver.mlir` prewalk lowers `name >>> type` to this tuple
  # before the nested `sequence` macro expands. Accept the original spelling as
  # well so the macro remains usable when expanded outside that prewalk.
  defp sequence_argument!({:>>>, _, [variable, type]}, index),
    do: sequence_argument!({variable, type}, index)

  defp sequence_argument!({variable = {name, _, context}, type}, index)
       when is_atom(name) and (is_atom(context) or is_nil(context)) do
    type_builder =
      quote do
        Deferred.defer(fn context -> Deferred.resolve(unquote(type), context) end)
      end

    binding =
      if name |> Atom.to_string() |> String.starts_with?("_") do
        quote do
          unquote(variable) = Block.get_arg!(sequence_block, unquote(index))
        end
      else
        quote do
          unquote(variable) = Block.get_arg!(sequence_block, unquote(index))
          _ = unquote(variable)
        end
      end

    {type_builder, binding}
  end

  defp sequence_argument!(invalid, _index),
    do: raise(ArgumentError, "sequence arguments must use `name >>> transform_type`, got: " <> Macro.to_string(invalid))

  @doc false
  def __build_sequence__(ctx, ip, name, type_builders, caller, body)
      when is_function(body, 2) do
    location = source_location(caller, ctx)
    argument_types = Enum.map(type_builders, &resolve_type!(&1, ctx, "sequence argument"))
    region = CAPI.mlirRegionCreate()

    try do
      block = Block.create(Enum.map(argument_types, &{&1, location}))
      CAPI.mlirRegionAppendOwnedBlock(region, block)
      body.(ctx, block)
      ensure_yield(block, ctx, location)

      readonly = Attribute.dictionary([{"transform.readonly", Attribute.unit(ctx: ctx)}], ctx: ctx)

      %Beaver.SSA{
        op: "transform.named_sequence",
        arguments: [
          region,
          sym_name: Attribute.string(name, ctx: ctx),
          function_type: Attribute.type(Type.function(argument_types, [], ctx: ctx)),
          arg_attrs: Attribute.array(List.duplicate(readonly, length(argument_types)), ctx: ctx),
          res_attrs: Attribute.array([], ctx: ctx)
        ],
        results: [],
        ctx: ctx,
        ip: ip,
        loc: location
      }
      |> Operation.create()
    rescue
      exception ->
        CAPI.mlirRegionDestroy(region)
        reraise exception, __STACKTRACE__
    catch
      kind, reason ->
        CAPI.mlirRegionDestroy(region)
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  @doc """
  Creates `transform.tune.knob` and returns its parameter handle.

  Integer, float, boolean, string, `:unit`, `MLIR.Attribute`, and deferred
  attribute values are accepted. Integer-only knobs default to
  `!transform.param<i64>`; other domains default to `!transform.any_param`.
  Pass `:type` to select an explicit Transform parameter type.
  """
  defmacro knob(name, options, opts \\ []),
    do: helper_ast(:__build_knob__, [name, options, opts], __CALLER__)

  @doc false
  def __build_knob__(ctx, ip, name, options, opts, caller) do
    name = runtime_name!(name, "knob")
    validate_knob_options!(options)
    validate_helper_options!(opts, [:type], "knob")
    location = source_location(caller, ctx)
    attributes = Enum.map(options, &knob_attribute!(&1, ctx))

    result_type =
      case Keyword.fetch(opts, :type) do
        {:ok, type} -> resolve_type!(type, ctx, "knob result")
        :error -> default_knob_type(options, ctx)
      end

    operation =
      create_operation(
        "transform.tune.knob",
        [name: Attribute.string(name, ctx: ctx), options: Attribute.array(attributes, ctx: ctx)],
        [result_type],
        ctx,
        ip,
        location
      )

    Operation.result(operation, 0)
  end

  defp validate_knob_options!(options) when is_list(options) and options != [], do: :ok

  defp validate_knob_options!([]), do: raise(ArgumentError, "knob options cannot be empty")

  defp validate_knob_options!(other),
    do: raise(ArgumentError, "knob options must be a non-empty list, got: #{inspect(other)}")

  defp default_knob_type(options, ctx),
    do: if(Enum.all?(options, &is_integer/1), do: param(Type.i64(ctx: ctx)), else: any_param(ctx: ctx))

  defp knob_attribute!(%Attribute{} = attribute, ctx) do
    ensure_same_context!(attribute, ctx, "knob option")
    attribute
  end

  defp knob_attribute!(value, ctx) when is_integer(value), do: Attribute.integer(Type.i64(ctx: ctx), value)

  defp knob_attribute!(value, ctx) when is_float(value), do: Attribute.float(Type.f64(ctx: ctx), value)

  defp knob_attribute!(value, ctx) when is_boolean(value), do: Attribute.bool(value, ctx: ctx)

  defp knob_attribute!(value, ctx) when is_binary(value), do: Attribute.string(value, ctx: ctx)

  defp knob_attribute!(:unit, ctx), do: Attribute.unit(ctx: ctx)

  defp knob_attribute!(%Deferred{} = deferred, ctx) do
    case Deferred.resolve(deferred, ctx) do
      %Attribute{} = attribute -> knob_attribute!(attribute, ctx)
      value -> unsupported_knob_option!(value)
    end
  end

  defp knob_attribute!(value, _context), do: unsupported_knob_option!(value)

  defp unsupported_knob_option!(value),
    do: raise(ArgumentError, "unsupported scalar or attribute in knob options: #{inspect(value)}")

  @doc """
  Packs one or more same-typed Transform parameter handles into one handle.

  This emits `transform.merge_handles` without deduplication, preserving the
  parameter order. It is useful with operations whose custom syntax accepts a
  runtime-sized parameter list, such as packed tile sizes or interchange for
  `transform.structured.tile_using_for`.

  The input handles are consumed by `transform.merge_handles`.
  """
  defmacro pack_params(params), do: helper_ast(:__pack_params__, [params], __CALLER__)

  @doc false
  def __pack_params__(ctx, ip, params, caller) do
    validate_packed_params!(params, ctx)
    result_type = params |> hd() |> Value.type()

    operation =
      create_operation(
        "transform.merge_handles",
        [handles: params],
        [result_type],
        ctx,
        ip,
        source_location(caller, ctx)
      )

    Operation.result(operation, 0)
  end

  defp validate_packed_params!(params, ctx) when is_list(params) and params != [] do
    unless Enum.all?(params, &match?(%Value{}, &1)),
      do: raise(ArgumentError, "pack_params expects a non-empty list of MLIR values")

    Enum.each(params, &ensure_same_context!(&1, ctx, "packed parameter"))
    [first | rest] = Enum.map(params, &Value.type/1)

    unless String.starts_with?(MLIR.to_string(first), "!transform.") and
             String.contains?(MLIR.to_string(first), "param"),
           do: raise(ArgumentError, "pack_params expects Transform parameter handles")

    unless Enum.all?(rest, &MLIR.equal?(&1, first)),
      do: raise(ArgumentError, "pack_params expects handles with the same Transform parameter type")
  end

  defp validate_packed_params!(_params, _context),
    do: raise(ArgumentError, "pack_params expects a non-empty list of MLIR values")

  @doc "Whether the linked LLVM provides `transform.memref.alloc_to_global`."
  def alloc_to_global_supported? do
    Code.ensure_loaded?(MLIR.Dialect.Transform) and
      function_exported?(MLIR.Dialect.Transform, :memref_alloc_to_global, 1)
  end

  @doc """
  Replaces the matched `memref.alloc` operations with globals.

  Returns handles for the inserted `memref.get_global` and `memref.global`
  operations, in that order.
  """
  defmacro alloc_to_global(target), do: helper_ast(:__alloc_to_global__, [target], __CALLER__)

  @doc false
  def __alloc_to_global__(ctx, ip, target, caller) do
    ensure_transform_helper_supported!(alloc_to_global_supported?(), "transform.memref.alloc_to_global")

    ensure_transform_handle!(target, ctx, "alloc_to_global target")

    operation =
      create_operation(
        "transform.memref.alloc_to_global",
        [alloc: target],
        [any_op(ctx: ctx), any_op(ctx: ctx)],
        ctx,
        ip,
        source_location(caller, ctx)
      )

    operation |> Operation.results() |> Enum.to_list()
  end

  @doc "Whether the linked LLVM provides `transform.loop.unroll_full`."
  def loop_unroll_full_supported?,
    do: Code.ensure_loaded?(MLIR.Dialect.Transform) and function_exported?(MLIR.Dialect.Transform, :loop_unroll_full, 1)

  @doc "Fully unrolls every `scf.for` or `affine.for` associated with `target`."
  defmacro loop_unroll_full(target), do: helper_ast(:__loop_unroll_full__, [target], __CALLER__)

  @doc false
  def __loop_unroll_full__(ctx, ip, target, caller) do
    ensure_transform_helper_supported!(loop_unroll_full_supported?(), "transform.loop.unroll_full")

    ensure_transform_handle!(target, ctx, "loop_unroll_full target")

    create_operation("transform.loop.unroll_full", [target: target], [], ctx, ip, source_location(caller, ctx))
  end

  @doc """
  Creates `transform.tune.alternatives` from explicit `branch` blocks.

      alternatives "vectorize" do
        branch do
          # Transform SSA operations
        end

        branch do
          # A second alternative
        end
      end

  Each branch receives an implicit `transform.yield` when it does not already
  end in one.
  """
  defmacro alternatives(name, branches_or_block) do
    caller = Macro.escape(__CALLER__)
    branches = branch_builders(branches_or_block)

    quote do
      DSL.__build_alternatives__(
        Beaver.Env.context(),
        Beaver.Env.ip(),
        unquote(name),
        unquote(branches),
        unquote(caller)
      )
    end
  end

  defp branch_builders(do: block), do: branch_builders_from_block!(block)
  defp branch_builders(other), do: other

  defp branch_builders_from_block!({:branch, _, [[do: body]]}), do: [branch_builder(body)]

  defp branch_builders_from_block!({:__block__, _, expressions}) do
    Enum.map(expressions, fn
      {:branch, _, [[do: body]]} ->
        branch_builder(body)

      invalid ->
        raise ArgumentError,
              "alternatives accepts only `branch do ... end` entries, got: " <>
                Macro.to_string(invalid)
    end)
  end

  defp branch_builders_from_block!(invalid),
    do: raise(ArgumentError, "alternatives expects `branch do ... end`, got: #{Macro.to_string(invalid)}")

  defp branch_builder(body) do
    quote do
      fn context, block ->
        Beaver.mlir ctx: context, ip: block do
          unquote(body)
        end
      end
    end
  end

  @doc false
  @spec __build_alternatives__(term(), term(), term(), term(), term()) :: term()
  def __build_alternatives__(ctx, ip, name, branches, caller) do
    name = runtime_name!(name, "alternatives")
    validate_branches!(branches)

    location = source_location(caller, ctx)
    regions = build_branch_regions(branches, ctx, location, [])

    try do
      %Beaver.SSA{
        op: "transform.tune.alternatives",
        arguments: [name: Attribute.string(name, ctx: ctx)],
        results: [],
        filler: fn -> regions end,
        ctx: ctx,
        ip: ip,
        loc: location
      }
      |> Operation.create()
    rescue
      exception ->
        Enum.each(regions, &CAPI.mlirRegionDestroy/1)
        reraise exception, __STACKTRACE__
    catch
      kind, reason ->
        Enum.each(regions, &CAPI.mlirRegionDestroy/1)
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  defp validate_branches!(branches) do
    unless is_list(branches) and branches != [] and Enum.all?(branches, &is_function(&1, 2)),
      do: raise(ArgumentError, "invalid branch layout for alternatives: expected one or more branch blocks")
  end

  defp build_branch_regions([], _context, _location, built), do: Enum.reverse(built)

  defp build_branch_regions([branch | rest], ctx, loc, built) do
    region = CAPI.mlirRegionCreate()

    region =
      try do
        block = Block.create()
        CAPI.mlirRegionAppendOwnedBlock(region, block)
        branch.(ctx, block)
        ensure_yield(block, ctx, loc)
        region
      rescue
        exception ->
          CAPI.mlirRegionDestroy(region)
          Enum.each(built, &CAPI.mlirRegionDestroy/1)
          reraise exception, __STACKTRACE__
      catch
        kind, reason ->
          CAPI.mlirRegionDestroy(region)
          Enum.each(built, &CAPI.mlirRegionDestroy/1)
          :erlang.raise(kind, reason, __STACKTRACE__)
      end

    # Keep the recursive call outside the cleanup scope above. A deeper
    # failure already destroys every region in `built`, including this one.
    build_branch_regions(rest, ctx, loc, [region | built])
  end

  @doc "Creates `transform.yield` with zero or more yielded handles."
  defmacro yield(operands \\ []), do: helper_ast(:__build_yield__, [operands], __CALLER__)

  @doc false
  def __build_yield__(ctx, ip, operands, caller),
    do: create_operation("transform.yield", List.wrap(operands), [], ctx, ip, source_location(caller, ctx))

  defp create_operation(name, arguments, results, ctx, ip, loc),
    do: %Beaver.SSA{op: name, arguments: arguments, results: results, ctx: ctx, ip: ip, loc: loc} |> Operation.create()

  defp ensure_yield(block, ctx, loc) do
    terminator = block |> Beaver.Walker.operations() |> Enum.to_list() |> List.last()

    if is_nil(terminator) or Operation.name(terminator) != "transform.yield",
      do: create_operation("transform.yield", [], [], ctx, block, loc)
  end

  @doc "Returns a deferred `!transform.any_op` type."
  def any_op(opts \\ []), do: Transform.any_op_type(opts)

  @doc "Returns a deferred `!transform.any_value` type."
  def any_value(opts \\ []), do: Transform.any_value_type(opts)

  @doc "Returns a deferred `!transform.any_param` type."
  def any_param(opts \\ []), do: Transform.any_param_type(opts)

  @doc "Returns a deferred operation-specific Transform handle type."
  def operation(name, opts \\ []), do: Transform.operation_type(name, opts)

  @doc "Returns a Transform parameter type wrapping an MLIR type."
  def param(type), do: Transform.param_type(type)

  defp resolve_type!(type, ctx, label) do
    case Deferred.resolve(type, ctx) do
      %Type{} = resolved ->
        ensure_same_context!(resolved, ctx, label)
        resolved

      value ->
        raise ArgumentError, "#{label} must resolve to an MLIR type, got: #{inspect(value)}"
    end
  end

  defp ensure_same_context!(entity, ctx, label) do
    unless MLIR.equal?(MLIR.context(entity), ctx),
      do: raise(ArgumentError, "#{label} belongs to a different MLIR context")
  end

  defp ensure_transform_handle!(%Value{} = value, ctx, label) do
    ensure_same_context!(value, ctx, label)

    unless value |> Value.type() |> MLIR.to_string() |> String.starts_with?("!transform."),
      do: raise(ArgumentError, "#{label} must be a Transform handle")

    value
  end

  defp ensure_transform_handle!(value, _context, label),
    do: raise(ArgumentError, "#{label} must be an MLIR value, got: #{inspect(value)}")

  defp ensure_transform_helper_supported!(true, _operation), do: :ok

  defp ensure_transform_helper_supported!(false, op), do: raise(ArgumentError, "linked LLVM does not support #{op}")

  defp validate_helper_options!(opts, supported, helper) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "#{helper} options must be a keyword list, got: #{inspect(opts)}")

    case Keyword.keys(opts) -- supported do
      [] -> :ok
      unsupported -> raise ArgumentError, "unsupported #{helper} options: #{inspect(unsupported)}"
    end
  end

  defp literal_name!(name, _label) when is_binary(name) and name != "", do: name
  defp literal_name!(name, label) when is_atom(name), do: runtime_name!(name, label)

  defp literal_name!(name, label),
    do: raise(ArgumentError, "#{label} name must be a literal non-empty string or atom, got: #{Macro.to_string(name)}")

  defp runtime_name!(name, _label) when is_binary(name) and name != "", do: name
  defp runtime_name!(name, _label) when is_atom(name), do: Atom.to_string(name)

  defp runtime_name!(name, label),
    do: raise(ArgumentError, "#{label} name must be a non-empty string or atom, got: #{inspect(name)}")

  defp source_location(caller, ctx),
    do: MLIR.Location.file(name: caller.file, line: caller.line, ctx: ctx)
end
