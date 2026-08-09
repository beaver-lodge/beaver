defmodule Beaver.MLIR.Dialect.Bufferization do
  @moduledoc """
  Operations and structured One-Shot Bufferization pipelines.

  The pipeline helpers validate and render pass options, preserve diagnostics
  through `Beaver.Composer`, and optionally append MLIR's ownership-based
  deallocation pipeline.
  """

  use Beaver.MLIR.Dialect,
    dialect: "bufferization",
    ops: Beaver.MLIR.Dialect.Registry.ops("bufferization")

  @boolean_options ~w(
    allow_return_allocs_from_loops
    allow_unknown_ops
    bufferize_function_boundaries
    check_parallel_regions
    copy_before_write
    must_infer_memory_space
    use_encoding_for_memory_space
  )a

  @scalar_options ~w(analysis_heuristic buffer_alignment)a
  @list_options ~w(dialect_filter no_analysis_func_filter)a
  @layout_options ~w(function_boundary_type_conversion unknown_type_conversion)a
  @control_options ~w(boundary_layout deallocate private_function_dynamic_ownership run)a
  @supported_options @boolean_options ++
                       @scalar_options ++ @list_options ++ @layout_options ++ @control_options

  @layout_values %{
    identity: "identity-layout-map",
    fully_dynamic: "fully-dynamic-layout-map",
    infer: "infer-layout-map"
  }

  @doc "Return a validated One-Shot Bufferize pass pipeline fragment."
  def one_shot_pipeline(opts \\ []) when is_list(opts) do
    validate_options!(opts)
    opts = expand_boundary_layout(opts)

    rendered =
      opts
      |> Keyword.take(@boolean_options ++ @scalar_options ++ @list_options ++ @layout_options)
      |> Enum.map_join(" ", &render_option/1)

    if rendered == "", do: "one-shot-bufferize", else: "one-shot-bufferize{#{rendered}}"
  end

  @doc "Return the complete bufferization pipeline as a `Beaver.Composer`."
  def pipeline(operation_or_module, opts \\ []) do
    composer = Beaver.Composer.append(operation_or_module, one_shot_pipeline(opts))

    if Keyword.get(opts, :deallocate, false) do
      deallocation =
        if Keyword.get(opts, :private_function_dynamic_ownership, false) do
          "buffer-deallocation-pipeline{private-function-dynamic-ownership=true}"
        else
          "buffer-deallocation-pipeline"
        end

      Beaver.Composer.append(composer, deallocation)
    else
      composer
    end
  end

  @doc "Run One-Shot Bufferization and return diagnostics without raising."
  def one_shot(operation_or_module, opts \\ []) do
    operation_or_module
    |> pipeline(opts)
    |> Beaver.Composer.run(Keyword.get(opts, :run, []))
  end

  @doc "Run One-Shot Bufferization, raising with formatted MLIR diagnostics on failure."
  def one_shot!(operation_or_module, opts \\ []) do
    operation_or_module
    |> pipeline(opts)
    |> Beaver.Composer.run!(Keyword.get(opts, :run, []))
  end

  defp validate_options!(opts) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "bufferization options must be a keyword list"
    end

    case Keyword.keys(opts) -- @supported_options do
      [] ->
        :ok

      [unsupported | _] ->
        raise ArgumentError, "unsupported bufferization option: #{inspect(unsupported)}"
    end

    Enum.each(
      Keyword.take(opts, @boolean_options ++ [:deallocate, :private_function_dynamic_ownership]),
      fn
        {_name, value} when is_boolean(value) -> :ok
        {name, value} -> raise ArgumentError, "#{name} must be a boolean, got: #{inspect(value)}"
      end
    )
  end

  defp expand_boundary_layout(opts) do
    case Keyword.pop(opts, :boundary_layout) do
      {nil, opts} ->
        opts

      {:identity, opts} ->
        opts
        |> Keyword.put_new(:unknown_type_conversion, :identity)
        |> Keyword.put_new(:function_boundary_type_conversion, :identity)

      {:fully_dynamic, opts} ->
        opts
        |> Keyword.put_new(:unknown_type_conversion, :fully_dynamic)
        |> Keyword.put_new(:function_boundary_type_conversion, :fully_dynamic)

      {:infer, opts} ->
        Keyword.put_new(opts, :function_boundary_type_conversion, :infer)

      {value, _opts} ->
        raise ArgumentError, "unsupported boundary layout: #{inspect(value)}"
    end
  end

  defp render_option({name, value}) when name in @boolean_options,
    do: "#{pass_name(name)}=#{value}"

  defp render_option({name, values}) when name in @list_options and is_list(values),
    do: "#{pass_name(name)}=#{Enum.map_join(values, ",", &to_string/1)}"

  defp render_option({name, value}) when name in @layout_options do
    case @layout_values do
      %{^value => spelling} -> "#{pass_name(name)}=#{spelling}"
      _ -> raise ArgumentError, "unsupported #{name}: #{inspect(value)}"
    end
  end

  defp render_option({:analysis_heuristic, value})
       when value in [:bottom_up, :top_down, :bottom_up_from_terminators],
       do: "analysis-heuristic=#{value |> to_string() |> String.replace("_", "-")}"

  defp render_option({:buffer_alignment, value}) when is_integer(value) and value > 0,
    do: "buffer-alignment=#{value}"

  defp render_option({name, value}),
    do: raise(ArgumentError, "invalid bufferization option #{name}: #{inspect(value)}")

  defp pass_name(name), do: name |> to_string() |> String.replace("_", "-")
end
