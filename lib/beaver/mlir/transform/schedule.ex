defmodule Beaver.MLIR.Transform.Schedule do
  @moduledoc """
  Deterministic discovery and resolution of Transform Tune schedules.

  A resolved schedule is immutable BEAM data: MLIR bytecode, printable text,
  the active choices, and a stable digest. Resolution writes choices back to
  `transform.tune.knob` and `transform.tune.alternatives`, so replay does not
  invoke the resolver or repeat a search.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.Transform

  @format 1
  @default_sequence "__transform_main"
  @default_max_candidates 10_000

  defmodule Option do
    @moduledoc "One enumerable option of a Tune choice."
    @enforce_keys [:index, :value, :mlir]
    defstruct [:index, :value, :mlir]

    @type t() :: %__MODULE__{index: non_neg_integer(), value: term(), mlir: String.t()}
  end

  defmodule Choice do
    @moduledoc "A discovered `transform.tune.knob` or alternatives operation."
    @enforce_keys [:name, :kind, :options, :path, :guards, :enumerable?]
    defstruct [:name, :kind, :options, :selected, :domain, :path, :guards, :enumerable?]

    @type kind() :: :knob | :alternatives
    @type guard() :: {String.t(), non_neg_integer()}

    @type t() :: %__MODULE__{
            name: String.t(),
            kind: kind(),
            options: [Option.t()],
            selected: term() | nil,
            domain: String.t() | nil,
            path: list(term()),
            guards: [guard()],
            enumerable?: boolean()
          }
  end

  defmodule Constraint do
    @moduledoc "An exported `transform.smt.constrain_params` operation."
    @enforce_keys [:ir, :path, :guards]
    defstruct [:ir, :path, :guards, operands: 0, results: 0]

    @type t() :: %__MODULE__{
            ir: String.t(),
            path: list(term()),
            guards: [Choice.guard()],
            operands: non_neg_integer(),
            results: non_neg_integer()
          }
  end

  defmodule Analysis do
    @moduledoc "Serializable discovery result for one named sequence."
    @enforce_keys [:sequence, :choices, :constraints]
    defstruct [:sequence, :choices, :constraints]

    @type t() :: %__MODULE__{
            sequence: String.t(),
            choices: [Choice.t()],
            constraints: [Constraint.t()]
          }
  end

  defmodule Resolved do
    @moduledoc "An immutable, cache-addressable, replayable Transform schedule."
    @enforce_keys [:format, :sequence, :bytecode, :text, :choices, :constraints, :digest]
    defstruct [
      :format,
      :sequence,
      :bytecode,
      :text,
      :choices,
      :constraints,
      :digest,
      :solver_metadata
    ]

    @type t() :: %__MODULE__{
            format: pos_integer(),
            sequence: String.t(),
            bytecode: binary(),
            text: String.t(),
            choices: map(),
            constraints: [Constraint.t()],
            digest: String.t(),
            solver_metadata: term()
          }
  end

  @type input() ::
          MLIR.Module.t()
          | MLIR.Operation.t()
          | binary()
          | {:text, binary()}
          | {:bytecode, binary()}
          | Resolved.t()

  @doc "Returns the entry-point symbol carried by a resolved schedule."
  def sequence(%Resolved{sequence: sequence}), do: sequence
  def sequence(_input), do: @default_sequence

  @doc "Discovers Tune choices and SMT constraints without executing the schedule."
  @spec analyze(input(), keyword()) :: {:ok, Analysis.t()} | {:error, Transform.Error.t()}
  def analyze(input, opts \\ []) do
    with_analysis_module(input, opts, fn module, sequence ->
      case analyze_module(module, sequence) do
        {:ok, analysis, _entries} -> {:ok, analysis}
        {:error, error} -> {:error, error}
      end
    end)
  end

  @doc "Returns exported SMT constraints even when no solver is configured."
  @spec constraints(input(), keyword()) ::
          {:ok, [Constraint.t()]} | {:error, Transform.Error.t()}
  def constraints(input, opts \\ []) do
    with {:ok, %Analysis{constraints: constraints}} <- analyze(input, opts) do
      {:ok, constraints}
    end
  end

  @doc """
  Enumerates active Tune choices in stable IR and option order.

  Existing selections remain fixed. Choices nested in an alternatives region
  are included only when that region is selected by the candidate. Arbitrary
  non-array knob domains remain inspectable but require an explicit resolver.
  """
  @spec enumerate(input(), keyword()) :: {:ok, [map()]} | {:error, Transform.Error.t()}
  def enumerate(input, opts \\ []) do
    max_candidates = Keyword.get(opts, :max_candidates, @default_max_candidates)

    if is_integer(max_candidates) and max_candidates > 0 do
      with {:ok, %Analysis{} = analysis} <- analyze(input, opts) do
        enumerate_choices(analysis.choices, max_candidates)
      end
    else
      {:error, invalid_error(":max_candidates must be a positive integer")}
    end
  end

  @doc "Resolves active Tune choices from a map, function, or resolver behaviour."
  @spec resolve(input(), map() | function() | module() | {module(), term()}, keyword()) ::
          {:ok, Resolved.t()} | {:error, Transform.Error.t()}
  def resolve(input, resolver, opts \\ []) do
    with_analysis_module(input, opts, fn module, sequence ->
      with {:ok, analysis, entries} <- analyze_module(module, sequence),
           {:ok, selections, _resolver_state} <- resolve_entries(entries, resolver),
           active_constraints <- active_constraints(analysis.constraints, selections),
           {:ok, solver_metadata} <-
             solve_constraints(active_constraints, selections, Keyword.get(opts, :solver)),
           :ok <- rewrite_entries(entries, selections, MLIR.context(module)) do
        bytecode = MLIR.Bytecode.write!(module)
        text = MLIR.to_string(module, generic: false)

        {:ok,
         %Resolved{
           format: @format,
           sequence: sequence,
           bytecode: bytecode,
           text: text,
           choices: selections,
           constraints: active_constraints,
           digest: digest(bytecode),
           solver_metadata: solver_metadata
         }}
      end
    end)
  end

  @doc "Bang variant of `resolve/3`."
  @spec resolve!(input(), map() | function() | module() | {module(), term()}, keyword()) ::
          Resolved.t()
  def resolve!(input, resolver, opts \\ []) do
    case resolve(input, resolver, opts) do
      {:ok, resolved} -> resolved
      {:error, error} -> raise error
    end
  end

  @doc "Serializes a resolved schedule as replayable MLIR bytecode or text."
  @spec serialize(Resolved.t(), :bytecode | :text) :: binary()
  def serialize(%Resolved{bytecode: bytecode}, :bytecode), do: bytecode
  def serialize(%Resolved{text: text}, :text), do: text

  @doc "Returns the stable identity included in incremental compilation cache keys."
  @spec cache_identity(Resolved.t() | binary()) :: tuple()
  def cache_identity(%Resolved{format: format, digest: digest, sequence: sequence}) do
    {:beaver_transform_schedule, format, sequence, digest}
  end

  def cache_identity(schedule) when is_binary(schedule) do
    {:beaver_transform_schedule, @format, @default_sequence, digest(schedule)}
  end

  @doc false
  def snapshot(input, opts \\ []) do
    with_analysis_module(input, opts, fn module, sequence ->
      {:ok, {MLIR.Bytecode.write!(module), sequence}}
    end)
  end

  @doc false
  def with_module(input, %MLIR.Context{} = context, fun) when is_function(fun, 1) do
    with {:ok, bytes} <- source_bytes(input),
         {:ok, module} <- parse_module(bytes, context) do
      try do
        case MLIR.verify(module) do
          {:ok, _module} -> fun.(module)
          :null -> {:error, invalid_error("parsed transform module is null")}
          {:error, diagnostics} -> {:error, invalid_error(:verification_failed, diagnostics)}
        end
      after
        MLIR.Module.destroy(module)
      end
    end
  rescue
    exception in [ArgumentError] ->
      {:error, invalid_error(Exception.message(exception))}
  end

  @doc false
  def find_sequence(%MLIR.Module{} = module, sequence) when is_binary(sequence) do
    module
    |> operations()
    |> Enum.find(fn operation ->
      MLIR.Operation.name(operation) == "transform.named_sequence" and
        attribute_value(operation, "sym_name") == sequence
    end)
    |> case do
      nil -> {:error, invalid_error({:sequence_not_found, sequence})}
      operation -> {:ok, operation}
    end
  end

  defp with_analysis_module(input, opts, fun) do
    sequence = Keyword.get(opts, :sequence, sequence(input))

    case input_context(input, opts) do
      {:borrowed, context} ->
        with_module(input, context, &fun.(&1, sequence))

      :owned ->
        context = MLIR.Context.create()

        try do
          with_module(input, context, &fun.(&1, sequence))
        after
          MLIR.Context.destroy(context)
        end
    end
  end

  defp input_context(input, opts) do
    case Keyword.get(opts, :ctx) do
      %MLIR.Context{} = context -> {:borrowed, context}
      nil -> source_context(input)
    end
  end

  defp source_context(%MLIR.Module{} = module), do: {:borrowed, MLIR.context(module)}
  defp source_context(%MLIR.Operation{} = operation), do: {:borrowed, MLIR.context(operation)}
  defp source_context(_input), do: :owned

  defp source_bytes(%Resolved{bytecode: bytecode}), do: {:ok, bytecode}

  defp source_bytes({format, bytes}) when format in [:text, :bytecode] and is_binary(bytes),
    do: {:ok, bytes}

  defp source_bytes(bytes) when is_binary(bytes), do: {:ok, bytes}
  defp source_bytes(%MLIR.Module{} = module), do: {:ok, MLIR.Bytecode.write!(module)}

  defp source_bytes(%MLIR.Operation{} = operation) do
    case enclosing_module(operation) do
      nil ->
        text =
          "module attributes {transform.with_named_sequence} {\n" <>
            MLIR.to_string(operation, generic: false) <> "\n}"

        {:ok, text}

      module_operation ->
        {:ok, MLIR.Bytecode.write!(module_operation)}
    end
  end

  defp source_bytes(input), do: {:error, invalid_error({:unsupported_schedule_input, input})}

  defp parse_module(bytes, context) do
    case MLIR.Module.create(bytes, ctx: context) do
      {:ok, module} -> {:ok, module}
      {:error, diagnostics} -> {:error, invalid_error(:parse_failed, diagnostics)}
    end
  end

  defp enclosing_module(operation) do
    if MLIR.Operation.name(operation) == "builtin.module",
      do: operation,
      else: enclosing_module_parent(MLIR.Operation.parent(operation))
  end

  defp enclosing_module_parent(operation) do
    cond do
      MLIR.null?(operation) -> nil
      MLIR.Operation.name(operation) == "builtin.module" -> operation
      true -> enclosing_module_parent(MLIR.Operation.parent(operation))
    end
  end

  defp analyze_module(module, sequence) do
    with {:ok, root} <- find_sequence(module, sequence) do
      {_operation, {choices, constraints, entries}} =
        scan_operation(root, [], [], {[], [], []})

      choices = Enum.reverse(choices)
      constraints = Enum.reverse(constraints)
      entries = Enum.reverse(entries)

      case duplicate_choice_names(choices) do
        [] ->
          {:ok, %Analysis{sequence: sequence, choices: choices, constraints: constraints},
           entries}

        duplicates ->
          {:error, invalid_error({:duplicate_choice_names, duplicates})}
      end
    end
  end

  defp scan_operation(operation, path, guards, {choices, constraints, entries} = acc) do
    case MLIR.Operation.name(operation) do
      "transform.tune.knob" ->
        {choice, option_attributes} = knob_choice(operation, path, guards)

        acc =
          {[choice | choices], constraints, [{choice, operation, option_attributes} | entries]}

        scan_children(operation, path, guards, acc)

      "transform.tune.alternatives" ->
        choice = alternatives_choice(operation, path, guards)
        acc = {[choice | choices], constraints, [{choice, operation, []} | entries]}
        scan_alternatives(operation, path, guards, choice.name, acc)

      "transform.smt.constrain_params" ->
        constraint = %Constraint{
          ir: MLIR.to_string(operation, generic: true),
          path: path,
          guards: guards,
          operands: Enum.count(Beaver.Walker.operands(operation)),
          results: Enum.count(Beaver.Walker.results(operation))
        }

        scan_children(operation, path, guards, {choices, [constraint | constraints], entries})

      _other ->
        scan_children(operation, path, guards, acc)
    end
  end

  defp scan_children(operation, path, guards, acc) do
    operation
    |> Beaver.Walker.regions()
    |> Enum.with_index()
    |> Enum.reduce(acc, fn {region, region_index}, acc ->
      scan_region(region, path ++ [{:region, region_index}], guards, acc)
    end)
    |> then(&{operation, &1})
  end

  defp scan_alternatives(operation, path, guards, choice_name, acc) do
    operation
    |> Beaver.Walker.regions()
    |> Enum.with_index()
    |> Enum.reduce(acc, fn {region, region_index}, acc ->
      scan_region(
        region,
        path ++ [{:region, region_index}],
        guards ++ [{choice_name, region_index}],
        acc
      )
    end)
    |> then(&{operation, &1})
  end

  defp scan_region(region, path, guards, acc) do
    region
    |> Beaver.Walker.blocks()
    |> Enum.with_index()
    |> Enum.reduce(acc, fn {block, block_index}, acc ->
      block
      |> Beaver.Walker.operations()
      |> Enum.with_index()
      |> Enum.reduce(acc, fn {operation, operation_index}, acc ->
        {_operation, acc} =
          scan_operation(
            operation,
            path ++ [{:block, block_index}, {:operation, operation_index}],
            guards,
            acc
          )

        acc
      end)
    end)
  end

  defp knob_choice(operation, path, guards) do
    name = required_attribute_value!(operation, "name")
    options_attribute = required_attribute!(operation, "options")
    selected_attribute = optional_attribute(operation, "selected")

    if MLIR.Attribute.array?(options_attribute) do
      option_attributes = Enum.to_list(options_attribute)

      options =
        option_attributes
        |> Enum.with_index()
        |> Enum.map(fn {attribute, index} -> option(attribute, index) end)

      selected = if selected_attribute, do: public_attribute_value(selected_attribute)

      {%Choice{
         name: name,
         kind: :knob,
         options: options,
         selected: selected,
         domain: MLIR.to_string(options_attribute),
         path: path,
         guards: guards,
         enumerable?: true
       }, option_attributes}
    else
      {%Choice{
         name: name,
         kind: :knob,
         options: [],
         selected: selected_attribute && public_attribute_value(selected_attribute),
         domain: MLIR.to_string(options_attribute),
         path: path,
         guards: guards,
         enumerable?: false
       }, []}
    end
  end

  defp alternatives_choice(operation, path, guards) do
    name = required_attribute_value!(operation, "name")
    count = Enum.count(Beaver.Walker.regions(operation))
    parameter_selected? = not Enum.empty?(Beaver.Walker.operands(operation))

    selected =
      cond do
        parameter_selected? ->
          :parameter

        attribute = optional_attribute(operation, "selected_region_attr") ->
          MLIR.Attribute.value(attribute)

        true ->
          nil
      end

    %Choice{
      name: name,
      kind: :alternatives,
      options:
        for(index <- 0..(count - 1), do: %Option{index: index, value: index, mlir: "#{index}"}),
      selected: selected,
      domain: "regions:#{count}",
      path: path,
      guards: guards,
      enumerable?: not parameter_selected?
    }
  end

  defp option(attribute, index) do
    %Option{
      index: index,
      value: public_attribute_value(attribute),
      mlir: MLIR.to_string(attribute)
    }
  end

  defp public_attribute_value(attribute) do
    value = MLIR.Attribute.value(attribute)
    if is_nil(value), do: MLIR.to_string(attribute), else: value
  end

  defp required_attribute!(operation, name) do
    case optional_attribute(operation, name) do
      nil -> raise ArgumentError, "#{MLIR.Operation.name(operation)} is missing #{name}"
      attribute -> attribute
    end
  end

  defp required_attribute_value!(operation, name) do
    operation |> required_attribute!(name) |> MLIR.Attribute.value()
  end

  defp optional_attribute(operation, name) do
    case MLIR.Operation.fetch(operation, name) do
      {:ok, attribute} -> attribute
      :error -> nil
    end
  end

  defp attribute_value(operation, name) do
    case optional_attribute(operation, name) do
      nil -> nil
      attribute -> MLIR.Attribute.value(attribute)
    end
  end

  defp duplicate_choice_names(choices) do
    choices
    |> Enum.frequencies_by(& &1.name)
    |> Enum.filter(fn {_name, count} -> count > 1 end)
    |> Enum.map(&elem(&1, 0))
    |> Enum.sort()
  end

  defp enumerate_choices(choices, max_candidates) do
    Enum.reduce_while(choices, {:ok, [%{}]}, fn choice, {:ok, candidates} ->
      enumerate_choice(choice, candidates, max_candidates)
    end)
  end

  defp enumerate_choice(%Choice{selected: :parameter}, candidates, _max_candidates),
    do: {:cont, {:ok, candidates}}

  defp enumerate_choice(choice, candidates, max_candidates) do
    cond do
      not active_for_any?(choice, candidates) ->
        {:cont, {:ok, candidates}}

      not choice.enumerable? and is_nil(choice.selected) ->
        error = invalid_error({:non_enumerable_choice, choice.name, choice.domain})
        {:halt, {:error, error}}

      true ->
        expand_choice(choice, candidates, max_candidates)
    end
  end

  defp expand_choice(choice, candidates, max_candidates) do
    values =
      if is_nil(choice.selected),
        do: Enum.map(choice.options, & &1.value),
        else: [choice.selected]

    candidates =
      Enum.flat_map(candidates, fn candidate ->
        if active?(choice.guards, candidate) do
          Enum.map(values, &Map.put(candidate, choice.name, &1))
        else
          [candidate]
        end
      end)

    if length(candidates) > max_candidates do
      error = invalid_error({:candidate_limit_exceeded, max_candidates, choice.name})
      {:halt, {:error, error}}
    else
      {:cont, {:ok, candidates}}
    end
  end

  defp active_for_any?(choice, candidates), do: Enum.any?(candidates, &active?(choice.guards, &1))

  defp active?(guards, selections) do
    Enum.all?(guards, fn {name, selected_region} ->
      Map.get(selections, name) == selected_region
    end)
  end

  defp resolve_entries(entries, resolver) do
    {resolver, state} = resolver_and_state(resolver)

    Enum.reduce_while(entries, {:ok, %{}, state}, fn entry, accumulator ->
      resolve_entry(entry, accumulator, resolver)
    end)
  end

  defp resolve_entry(
         {%Choice{selected: :parameter}, _operation, _attributes},
         {:ok, selections, state},
         _resolver
       ),
       do: {:cont, {:ok, selections, state}}

  defp resolve_entry({choice, _operation, _attributes}, {:ok, selections, state}, resolver) do
    if active?(choice.guards, selections) do
      resolve_active_choice(choice, selections, resolver, state)
    else
      {:cont, {:ok, selections, state}}
    end
  end

  defp resolve_active_choice(choice, selections, resolver, state) do
    case resolve_choice(resolver, choice, state) do
      {:ok, value, state} ->
        record_selection(choice, value, selections, state)

      {:unresolved, state} when not is_nil(choice.selected) ->
        {:cont, {:ok, Map.put(selections, choice.name, choice.selected), state}}

      {:unresolved, _state} ->
        {:halt, {:error, invalid_error({:unresolved_choice, choice.name})}}

      {:error, reason, _state} ->
        {:halt, {:error, invalid_error({:resolver_failed, choice.name, reason})}}
    end
  rescue
    exception ->
      reason = {:resolver_exception, choice.name, Exception.message(exception)}
      {:halt, {:error, invalid_error(reason)}}
  catch
    kind, reason ->
      reason = {:resolver_failure, choice.name, kind, reason}
      {:halt, {:error, invalid_error(reason)}}
  end

  defp record_selection(choice, value, selections, state) do
    case validate_selection(choice, value) do
      {:ok, normalized} ->
        {:cont, {:ok, Map.put(selections, choice.name, normalized), state}}

      {:error, reason} ->
        {:halt, {:error, invalid_error(reason)}}
    end
  end

  defp resolver_and_state({module, state}) when is_atom(module), do: {{:module, module}, state}
  defp resolver_and_state(module) when is_atom(module), do: {{:module, module}, nil}
  defp resolver_and_state(resolver), do: {resolver, nil}

  defp resolve_choice(resolver, choice, state) when is_map(resolver) do
    case map_fetch_choice(resolver, choice.name) do
      {:ok, value} -> {:ok, value, state}
      :error -> {:unresolved, state}
    end
  end

  defp resolve_choice(resolver, choice, state) when is_function(resolver, 1) do
    normalize_resolver_result(resolver.(choice), state)
  end

  defp resolve_choice(resolver, choice, state) when is_function(resolver, 2) do
    normalize_resolver_result(resolver.(choice, state), state)
  end

  defp resolve_choice({:module, module}, choice, state) do
    normalize_resolver_result(module.resolve(choice, state), state)
  end

  defp normalize_resolver_result({:ok, value, state}, _old_state), do: {:ok, value, state}
  defp normalize_resolver_result({:ok, value}, state), do: {:ok, value, state}
  defp normalize_resolver_result({:error, reason, state}, _old_state), do: {:error, reason, state}
  defp normalize_resolver_result({:error, reason}, state), do: {:error, reason, state}
  defp normalize_resolver_result({:unresolved, state}, _old_state), do: {:unresolved, state}
  defp normalize_resolver_result(:unresolved, state), do: {:unresolved, state}
  defp normalize_resolver_result(value, state), do: {:ok, value, state}

  defp map_fetch_choice(map, name) do
    case Map.fetch(map, name) do
      {:ok, value} ->
        {:ok, value}

      :error ->
        map_fetch_atom_choice(map, name)
    end
  end

  defp map_fetch_atom_choice(map, name) do
    Enum.find_value(map, :error, fn
      {key, value} when is_atom(key) -> atom_choice(key, value, name)
      _entry -> nil
    end)
  end

  defp atom_choice(key, value, name) do
    if Atom.to_string(key) == name, do: {:ok, value}
  end

  defp validate_selection(%Choice{kind: :alternatives, options: options, name: name}, value) do
    value = if match?(%Option{}, value), do: value.value, else: value

    if is_integer(value) and Enum.any?(options, &(&1.value == value)) do
      {:ok, value}
    else
      {:error, {:invalid_alternative, name, value}}
    end
  end

  defp validate_selection(%Choice{kind: :knob, enumerable?: true} = choice, {:option, index})
       when is_integer(index) do
    case Enum.at(choice.options, index) do
      nil -> {:error, {:invalid_knob_option, choice.name, index}}
      option -> {:ok, option.value}
    end
  end

  defp validate_selection(%Choice{kind: :knob, enumerable?: true} = choice, %Option{} = option) do
    validate_selection(choice, {:option, option.index})
  end

  defp validate_selection(%Choice{kind: :knob, enumerable?: true} = choice, value) do
    case Enum.find(choice.options, fn option -> option.value == value or option.mlir == value end) do
      nil -> {:error, {:invalid_knob_option, choice.name, value}}
      option -> {:ok, option.value}
    end
  end

  defp validate_selection(%Choice{kind: :knob, enumerable?: false}, value)
       when is_binary(value),
       do: {:ok, value}

  defp validate_selection(%Choice{name: name}, value),
    do: {:error, {:invalid_knob_option, name, value}}

  defp rewrite_entries(entries, selections, context) do
    Enum.reduce_while(entries, :ok, fn entry, :ok ->
      rewrite_entry(entry, selections, context)
    end)
  end

  defp rewrite_entry({choice, operation, option_attributes}, selections, context) do
    if active?(choice.guards, selections) and Map.has_key?(selections, choice.name) do
      value = Map.fetch!(selections, choice.name)

      case selection_attribute(choice, value, option_attributes, context) do
        {:ok, attribute_name, attribute} ->
          MLIR.CAPI.mlirOperationSetAttributeByName(
            operation,
            MLIR.StringRef.create(attribute_name),
            attribute
          )

          {:cont, :ok}

        {:error, reason} ->
          {:halt, {:error, invalid_error(reason)}}
      end
    else
      {:cont, :ok}
    end
  end

  defp selection_attribute(%Choice{kind: :alternatives}, value, _options, context) do
    {:ok, "selected_region_attr", MLIR.Attribute.integer(MLIR.Type.i64(ctx: context), value)}
  end

  defp selection_attribute(%Choice{kind: :knob, enumerable?: true} = choice, value, attrs, _ctx) do
    case Enum.find_index(choice.options, fn option ->
           option.value == value or option.mlir == value
         end) do
      nil -> {:error, {:invalid_knob_option, choice.name, value}}
      index -> {:ok, "selected", Enum.at(attrs, index)}
    end
  end

  defp selection_attribute(%Choice{kind: :knob, enumerable?: false}, value, _attrs, context) do
    {:ok, "selected", MLIR.Attribute.get(value, ctx: context)}
  rescue
    exception in [ArgumentError] ->
      {:error, {:invalid_attribute, value, Exception.message(exception)}}
  end

  defp active_constraints(constraints, selections) do
    Enum.filter(constraints, &active?(&1.guards, selections))
  end

  defp solve_constraints([], _selections, _solver), do: {:ok, nil}
  defp solve_constraints(_constraints, _selections, nil), do: {:ok, nil}

  defp solve_constraints(constraints, selections, solver) do
    {solver, state} = solver_and_state(solver)

    solver
    |> call_solver(constraints, selections, state)
    |> normalize_solver_result()
  rescue
    exception -> {:error, constraint_error({:solver_exception, Exception.message(exception)})}
  catch
    kind, reason -> {:error, constraint_error({:solver_failure, kind, reason})}
  end

  defp call_solver(fun, constraints, selections, _state) when is_function(fun, 2),
    do: fun.(constraints, selections)

  defp call_solver(fun, constraints, selections, state) when is_function(fun, 3),
    do: fun.(constraints, selections, state)

  defp call_solver({:module, module}, constraints, selections, state),
    do: module.solve(constraints, selections, state)

  defp normalize_solver_result(:ok), do: {:ok, nil}
  defp normalize_solver_result({:ok, metadata}), do: {:ok, metadata}
  defp normalize_solver_result({:ok, metadata, _new_state}), do: {:ok, metadata}
  defp normalize_solver_result({:error, reason}), do: {:error, constraint_error(reason)}

  defp normalize_solver_result({:error, reason, _new_state}),
    do: {:error, constraint_error(reason)}

  defp normalize_solver_result(other),
    do: {:error, constraint_error({:invalid_solver_result, other})}

  defp solver_and_state({module, state}) when is_atom(module), do: {{:module, module}, state}
  defp solver_and_state(module) when is_atom(module), do: {{:module, module}, nil}
  defp solver_and_state(solver), do: {solver, nil}

  defp operations(module) do
    {_module, operations} =
      Beaver.Walker.prewalk(module, [], fn
        %MLIR.Operation{} = operation, acc -> {operation, [operation | acc]}
        entity, acc -> {entity, acc}
      end)

    Enum.reverse(operations)
  end

  defp invalid_error(reason, diagnostics \\ []) do
    %Transform.Error{
      kind: :invalid_schedule,
      reason: reason,
      diagnostics: MLIR.Diagnostic.process(diagnostics)
    }
  end

  defp constraint_error(reason) do
    %Transform.Error{kind: :constraint_failure, reason: reason}
  end

  defp digest(bytes) do
    :crypto.hash(:sha256, bytes) |> Base.encode16(case: :lower)
  end
end
