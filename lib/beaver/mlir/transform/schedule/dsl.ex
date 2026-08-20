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

  defp schedule_function_name!(other) do
    raise ArgumentError,
          "defschedule expects a zero-arity function name, got: #{Macro.to_string(other)}"
  end

  @doc false
  def __build_schedule__(opts, caller, build_body) when is_function(build_body, 2) do
    context = schedule_context!(opts)
    location = source_location(caller, context)
    module = MLIR.Module.empty(location)

    try do
      module
      |> MLIR.Operation.from_module()
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
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "schedule options must be a keyword list, got: #{inspect(opts)}"
    end

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

  defp put_unit_attribute(operation, name, context) do
    MLIR.CAPI.mlirOperationSetAttributeByName(
      operation,
      MLIR.StringRef.create(name),
      MLIR.Attribute.unit(ctx: context)
    )

    operation
  end

  defp ensure_named_sequence!(module) do
    found? =
      module
      |> MLIR.Module.body()
      |> Beaver.Walker.operations()
      |> Enum.any?(&(MLIR.Operation.name(&1) == "transform.named_sequence"))

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
  defmacro sequence(do: body) do
    build_sequence_ast("__transform_main", default_root_argument(), body, __CALLER__)
  end

  @doc "Declares a named sequence with one default `any_op` root handle."
  defmacro sequence(name, do: body) do
    build_sequence_ast(literal_name!(name, "sequence"), default_root_argument(), body, __CALLER__)
  end

  @doc """
  Declares a named sequence with typed handle arguments.

  Arguments use Beaver's `>>>` spelling, for example
  `[root >>> any_op(), function >>> operation("func.func")]`. Sequence results
  are intentionally expressed by ordinary Transform SSA operations and
  `yield/1`; this helper currently declares result-free entry points.
  """
  defmacro sequence(name, arguments, do: body) do
    build_sequence_ast(literal_name!(name, "sequence"), arguments, body, __CALLER__)
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

  defp sequence_arguments!(other) do
    raise ArgumentError,
          "sequence arguments must be a literal list, got: #{Macro.to_string(other)}"
  end

  # The surrounding `Beaver.mlir` prewalk lowers `name >>> type` to this tuple
  # before the nested `sequence` macro expands. Accept the original spelling as
  # well so the macro remains usable when expanded outside that prewalk.
  defp sequence_argument!({:>>>, _, [variable, type]}, index),
    do: sequence_argument!({variable, type}, index)

  defp sequence_argument!({variable = {name, _, context}, type}, index)
       when is_atom(name) and (is_atom(context) or is_nil(context)) do
    type_builder =
      quote do
        Beaver.Deferred.defer(fn context ->
          Beaver.Deferred.resolve(unquote(type), context)
        end)
      end

    binding =
      if name |> Atom.to_string() |> String.starts_with?("_") do
        quote do
          unquote(variable) = Beaver.MLIR.Block.get_arg!(sequence_block, unquote(index))
        end
      else
        quote do
          unquote(variable) = Beaver.MLIR.Block.get_arg!(sequence_block, unquote(index))
          _ = unquote(variable)
        end
      end

    {type_builder, binding}
  end

  defp sequence_argument!(invalid, _index) do
    raise ArgumentError,
          "sequence arguments must use `name >>> transform_type`, got: " <>
            Macro.to_string(invalid)
  end

  @doc false
  def __build_sequence__(context, insertion_point, name, type_builders, caller, body)
      when is_function(body, 2) do
    location = source_location(caller, context)
    argument_types = Enum.map(type_builders, &resolve_type!(&1, context, "sequence argument"))
    region = MLIR.CAPI.mlirRegionCreate()

    try do
      block = MLIR.Block.create(Enum.map(argument_types, &{&1, location}))
      MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)
      body.(context, block)
      ensure_yield(block, context, location)

      readonly =
        MLIR.Attribute.dictionary(
          [{"transform.readonly", MLIR.Attribute.unit(ctx: context)}],
          ctx: context
        )

      %Beaver.SSA{
        op: "transform.named_sequence",
        arguments: [
          region,
          sym_name: MLIR.Attribute.string(name, ctx: context),
          function_type:
            MLIR.Attribute.type(MLIR.Type.function(argument_types, [], ctx: context)),
          arg_attrs:
            MLIR.Attribute.array(List.duplicate(readonly, length(argument_types)), ctx: context),
          res_attrs: MLIR.Attribute.array([], ctx: context)
        ],
        results: [],
        ctx: context,
        ip: insertion_point,
        loc: location
      }
      |> MLIR.Operation.create()
    rescue
      exception ->
        MLIR.CAPI.mlirRegionDestroy(region)
        reraise exception, __STACKTRACE__
    catch
      kind, reason ->
        MLIR.CAPI.mlirRegionDestroy(region)
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
  defmacro knob(name, options, opts \\ []) do
    caller = Macro.escape(__CALLER__)

    quote do
      DSL.__build_knob__(
        Beaver.Env.context(),
        Beaver.Env.ip(),
        unquote(name),
        unquote(options),
        unquote(opts),
        unquote(caller)
      )
    end
  end

  @doc false
  def __build_knob__(context, insertion_point, name, options, opts, caller) do
    name = runtime_name!(name, "knob")
    validate_knob_options!(options)
    validate_helper_options!(opts, [:type], "knob")
    location = source_location(caller, context)
    attributes = Enum.map(options, &knob_attribute!(&1, context))

    result_type =
      case Keyword.fetch(opts, :type) do
        {:ok, type} -> resolve_type!(type, context, "knob result")
        :error -> default_knob_type(options, context)
      end

    operation =
      %Beaver.SSA{
        op: "transform.tune.knob",
        arguments: [
          name: MLIR.Attribute.string(name, ctx: context),
          options: MLIR.Attribute.array(attributes, ctx: context)
        ],
        results: [result_type],
        ctx: context,
        ip: insertion_point,
        loc: location
      }
      |> MLIR.Operation.create()

    MLIR.Operation.result(operation, 0)
  end

  defp validate_knob_options!(options) when is_list(options) and options != [], do: :ok

  defp validate_knob_options!([]), do: raise(ArgumentError, "knob options cannot be empty")

  defp validate_knob_options!(other) do
    raise ArgumentError, "knob options must be a non-empty list, got: #{inspect(other)}"
  end

  defp default_knob_type(options, context) do
    if Enum.all?(options, &is_integer/1) do
      param(MLIR.Type.i64(ctx: context))
    else
      any_param(ctx: context)
    end
  end

  defp knob_attribute!(%MLIR.Attribute{} = attribute, context) do
    ensure_same_context!(attribute, context, "knob option")
    attribute
  end

  defp knob_attribute!(value, context) when is_integer(value),
    do: MLIR.Attribute.integer(MLIR.Type.i64(ctx: context), value)

  defp knob_attribute!(value, context) when is_float(value),
    do: MLIR.Attribute.float(MLIR.Type.f64(ctx: context), value)

  defp knob_attribute!(value, context) when is_boolean(value),
    do: MLIR.Attribute.bool(value, ctx: context)

  defp knob_attribute!(value, context) when is_binary(value),
    do: MLIR.Attribute.string(value, ctx: context)

  defp knob_attribute!(:unit, context), do: MLIR.Attribute.unit(ctx: context)

  defp knob_attribute!(%Beaver.Deferred{} = deferred, context) do
    case Beaver.Deferred.resolve(deferred, context) do
      %MLIR.Attribute{} = attribute -> knob_attribute!(attribute, context)
      value -> unsupported_knob_option!(value)
    end
  end

  defp knob_attribute!(value, _context), do: unsupported_knob_option!(value)

  defp unsupported_knob_option!(value) do
    raise ArgumentError, "unsupported scalar or attribute in knob options: #{inspect(value)}"
  end

  @doc """
  Packs one or more same-typed Transform parameter handles into one handle.

  This emits `transform.merge_handles` without deduplication, preserving the
  parameter order. It is useful with operations whose custom syntax accepts a
  runtime-sized parameter list, such as packed tile sizes or interchange for
  `transform.structured.tile_using_for`.

  The input handles are consumed by `transform.merge_handles`.
  """
  defmacro pack_params(params) do
    caller = Macro.escape(__CALLER__)

    quote do
      DSL.__pack_params__(
        Beaver.Env.context(),
        Beaver.Env.ip(),
        unquote(params),
        unquote(caller)
      )
    end
  end

  @doc false
  def __pack_params__(context, insertion_point, params, caller) do
    validate_packed_params!(params, context)
    result_type = params |> hd() |> MLIR.Value.type()

    operation =
      %Beaver.SSA{
        op: "transform.merge_handles",
        arguments: [handles: params],
        results: [result_type],
        ctx: context,
        ip: insertion_point,
        loc: source_location(caller, context)
      }
      |> MLIR.Operation.create()

    MLIR.Operation.result(operation, 0)
  end

  defp validate_packed_params!(params, context) when is_list(params) and params != [] do
    unless Enum.all?(params, &match?(%MLIR.Value{}, &1)) do
      raise ArgumentError, "pack_params expects a non-empty list of MLIR values"
    end

    Enum.each(params, &ensure_same_context!(&1, context, "packed parameter"))
    [first | rest] = Enum.map(params, &MLIR.Value.type/1)

    unless String.starts_with?(MLIR.to_string(first), "!transform.") and
             String.contains?(MLIR.to_string(first), "param") do
      raise ArgumentError, "pack_params expects Transform parameter handles"
    end

    unless Enum.all?(rest, &MLIR.equal?(&1, first)) do
      raise ArgumentError, "pack_params expects handles with the same Transform parameter type"
    end
  end

  defp validate_packed_params!(_params, _context) do
    raise ArgumentError, "pack_params expects a non-empty list of MLIR values"
  end

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
  defmacro alloc_to_global(target) do
    caller = Macro.escape(__CALLER__)

    quote do
      DSL.__alloc_to_global__(
        Beaver.Env.context(),
        Beaver.Env.ip(),
        unquote(target),
        unquote(caller)
      )
    end
  end

  @doc false
  def __alloc_to_global__(context, insertion_point, target, caller) do
    ensure_transform_helper_supported!(
      alloc_to_global_supported?(),
      "transform.memref.alloc_to_global"
    )

    ensure_transform_handle!(target, context, "alloc_to_global target")

    operation =
      %Beaver.SSA{
        op: "transform.memref.alloc_to_global",
        arguments: [alloc: target],
        results: [any_op(ctx: context), any_op(ctx: context)],
        ctx: context,
        ip: insertion_point,
        loc: source_location(caller, context)
      }
      |> MLIR.Operation.create()

    operation |> MLIR.Operation.results() |> Enum.to_list()
  end

  @doc "Whether the linked LLVM provides `transform.loop.unroll_full`."
  def loop_unroll_full_supported? do
    Code.ensure_loaded?(MLIR.Dialect.Transform) and
      function_exported?(MLIR.Dialect.Transform, :loop_unroll_full, 1)
  end

  @doc "Fully unrolls every `scf.for` or `affine.for` associated with `target`."
  defmacro loop_unroll_full(target) do
    caller = Macro.escape(__CALLER__)

    quote do
      DSL.__loop_unroll_full__(
        Beaver.Env.context(),
        Beaver.Env.ip(),
        unquote(target),
        unquote(caller)
      )
    end
  end

  @doc false
  def __loop_unroll_full__(context, insertion_point, target, caller) do
    ensure_transform_helper_supported!(
      loop_unroll_full_supported?(),
      "transform.loop.unroll_full"
    )

    ensure_transform_handle!(target, context, "loop_unroll_full target")

    %Beaver.SSA{
      op: "transform.loop.unroll_full",
      arguments: [target: target],
      results: [],
      ctx: context,
      ip: insertion_point,
      loc: source_location(caller, context)
    }
    |> MLIR.Operation.create()
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

  defp branch_builders_from_block!({:branch, _, [[do: body]]}),
    do: [branch_builder(body)]

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

  defp branch_builders_from_block!(invalid) do
    raise ArgumentError,
          "alternatives expects `branch do ... end`, got: #{Macro.to_string(invalid)}"
  end

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
  def __build_alternatives__(context, insertion_point, name, branches, caller) do
    name = runtime_name!(name, "alternatives")
    validate_branches!(branches)

    location = source_location(caller, context)
    regions = build_branch_regions(branches, context, location, [])

    try do
      %Beaver.SSA{
        op: "transform.tune.alternatives",
        arguments: [name: MLIR.Attribute.string(name, ctx: context)],
        results: [],
        filler: fn -> regions end,
        ctx: context,
        ip: insertion_point,
        loc: location
      }
      |> MLIR.Operation.create()
    rescue
      exception ->
        Enum.each(regions, &MLIR.CAPI.mlirRegionDestroy/1)
        reraise exception, __STACKTRACE__
    catch
      kind, reason ->
        Enum.each(regions, &MLIR.CAPI.mlirRegionDestroy/1)
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  defp validate_branches!(branches) do
    unless is_list(branches) and branches != [] and Enum.all?(branches, &is_function(&1, 2)) do
      raise ArgumentError,
            "invalid branch layout for alternatives: expected one or more branch blocks"
    end
  end

  defp build_branch_regions([], _context, _location, built), do: Enum.reverse(built)

  defp build_branch_regions([branch | rest], context, location, built) do
    region = MLIR.CAPI.mlirRegionCreate()

    region =
      try do
        block = MLIR.Block.create()
        MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)
        branch.(context, block)
        ensure_yield(block, context, location)
        region
      rescue
        exception ->
          MLIR.CAPI.mlirRegionDestroy(region)
          Enum.each(built, &MLIR.CAPI.mlirRegionDestroy/1)
          reraise exception, __STACKTRACE__
      catch
        kind, reason ->
          MLIR.CAPI.mlirRegionDestroy(region)
          Enum.each(built, &MLIR.CAPI.mlirRegionDestroy/1)
          :erlang.raise(kind, reason, __STACKTRACE__)
      end

    # Keep the recursive call outside the cleanup scope above. A deeper
    # failure already destroys every region in `built`, including this one.
    build_branch_regions(rest, context, location, [region | built])
  end

  @doc "Creates `transform.yield` with zero or more yielded handles."
  defmacro yield(operands \\ []) do
    caller = Macro.escape(__CALLER__)

    quote do
      DSL.__build_yield__(
        Beaver.Env.context(),
        Beaver.Env.ip(),
        unquote(operands),
        unquote(caller)
      )
    end
  end

  @doc false
  def __build_yield__(context, insertion_point, operands, caller) do
    %Beaver.SSA{
      op: "transform.yield",
      arguments: List.wrap(operands),
      results: [],
      ctx: context,
      ip: insertion_point,
      loc: source_location(caller, context)
    }
    |> MLIR.Operation.create()
  end

  defp ensure_yield(block, context, location) do
    terminator = block |> Beaver.Walker.operations() |> Enum.to_list() |> List.last()

    if is_nil(terminator) or MLIR.Operation.name(terminator) != "transform.yield" do
      %Beaver.SSA{
        op: "transform.yield",
        arguments: [],
        results: [],
        ctx: context,
        ip: block,
        loc: location
      }
      |> MLIR.Operation.create()
    end
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

  defp resolve_type!(type, context, label) do
    case Beaver.Deferred.resolve(type, context) do
      %MLIR.Type{} = resolved ->
        ensure_same_context!(resolved, context, label)
        resolved

      value ->
        raise ArgumentError, "#{label} must resolve to an MLIR type, got: #{inspect(value)}"
    end
  end

  defp ensure_same_context!(entity, context, label) do
    unless MLIR.equal?(MLIR.context(entity), context) do
      raise ArgumentError, "#{label} belongs to a different MLIR context"
    end
  end

  defp ensure_transform_handle!(%MLIR.Value{} = value, context, label) do
    ensure_same_context!(value, context, label)

    unless value |> MLIR.Value.type() |> MLIR.to_string() |> String.starts_with?("!transform.") do
      raise ArgumentError, "#{label} must be a Transform handle"
    end

    value
  end

  defp ensure_transform_handle!(value, _context, label) do
    raise ArgumentError, "#{label} must be an MLIR value, got: #{inspect(value)}"
  end

  defp ensure_transform_helper_supported!(true, _operation), do: :ok

  defp ensure_transform_helper_supported!(false, operation) do
    raise ArgumentError, "linked LLVM does not support #{operation}"
  end

  defp validate_helper_options!(opts, supported, helper) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "#{helper} options must be a keyword list, got: #{inspect(opts)}"
    end

    case Keyword.keys(opts) -- supported do
      [] -> :ok
      unsupported -> raise ArgumentError, "unsupported #{helper} options: #{inspect(unsupported)}"
    end
  end

  defp literal_name!(name, _label) when is_binary(name) and name != "", do: name
  defp literal_name!(name, label) when is_atom(name), do: runtime_name!(name, label)

  defp literal_name!(name, label) do
    raise ArgumentError,
          "#{label} name must be a literal non-empty string or atom, got: #{Macro.to_string(name)}"
  end

  defp runtime_name!(name, _label) when is_binary(name) and name != "", do: name
  defp runtime_name!(name, _label) when is_atom(name), do: Atom.to_string(name)

  defp runtime_name!(name, label) do
    raise ArgumentError, "#{label} name must be a non-empty string or atom, got: #{inspect(name)}"
  end

  defp source_location(caller, context) do
    MLIR.Location.file(name: caller.file, line: caller.line, ctx: context)
  end
end
