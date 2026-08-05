defmodule Beaver.MLIR.ODS.Dump do
  @moduledoc """
  This module provides functionality for working with MLIR ODS (Operation Definition Specification) dumps.
  It allows looking up operation definitions by name and generating documentation for them.

  The module loads ODS dump data on demand and provides functions to:
  - Look up operations by their fully qualified names (e.g. "affine.for")
  - Generate documentation for operations including their attributes, operands, and results
  - Check if an operation supports result type inference

  Operations can be looked up using `lookup/1`, and documentation can be generated using `gen_doc/1`.
  """
  @cache_key {__MODULE__, :operations}
  @external_resource Application.app_dir(:beaver, "priv/generated/ods_dump.json")

  @on_load :clear_cache

  @doc false
  def clear_cache do
    :persistent_term.erase(@cache_key)
    :ok
  end

  @doc """
  Lookup an operation by name (e.g. "affine.for").
  """
  def lookup(op) do
    case Map.fetch(operations(), op) do
      {:ok, found} -> {:ok, found}
      :error -> {:error, "failed to find ODS dump of #{inspect(op)}"}
    end
  end

  defp operations do
    case :persistent_term.get(@cache_key, :not_loaded) do
      :not_loaded -> load_operations()
      operations -> operations
    end
  end

  defp load_operations do
    operations =
      :beaver
      |> Application.app_dir("priv/generated/ods_dump.json")
      |> File.read!()
      |> Jason.decode!()
      |> Map.fetch!("dialects")
      |> Enum.flat_map(&Map.fetch!(&1, "operations"))
      |> Map.new(fn %{"name" => name} = op ->
        validate_operand_names!(op)
        {name, op}
      end)

    :persistent_term.put(@cache_key, operations)
    operations
  end

  defp validate_operand_names!(%{"name" => name} = op) do
    # Give anonymous operands temporary sequential names so multiple anonymous
    # operands are not mistaken for duplicate named operands.
    {operands, _} =
      Enum.map_reduce(op["operands"] || [], 0, fn operand, index ->
        if operand["name"] == "" do
          {Map.put(operand, "name", "arg#{index}"), index + 1}
        else
          {operand, index}
        end
      end)

    duplicates =
      operands
      |> Enum.map(& &1["name"])
      |> Enum.frequencies()
      |> Enum.filter(fn {_operand_name, count} -> count > 1 end)
      |> Enum.map(&elem(&1, 0))

    if duplicates != [] do
      raise "Duplicate operand names found in ODS dump for operation '#{name}': #{inspect(duplicates)}"
    end
  end

  defp fmt_constraint(constraint) do
    if String.contains?(constraint, "anonymous") do
      "anonymous/composite constraint"
    else
      "`#{constraint}`"
    end
  end

  defp fmt_name(""), do: "anonymous"
  defp fmt_name(name), do: "`#{name}`"

  defp gen_if_exist(op, key) do
    decls = op[key]

    if decls do
      """

      ## #{String.capitalize(key)}
      #{Enum.map_join(decls, "\n", &"- #{fmt_name(&1["name"])} - #{&1["kind"]}, #{fmt_constraint(&1["constraint"])}, #{&1["description"]}")}
      """
    else
      ""
    end
  end

  @doc false

  def gen_doc(
        %{
          "name" => name
        } = op
      ) do
    summary = op["summary"]

    description = op["description"]

    description =
      Regex.replace(~r{\(\.\./(.+?)\.md}, description, "(https://mlir.llvm.org/docs/\\1")

    description =
      Regex.replace(~r{\(\.\./(.+?)\/\#}, description, "(https://mlir.llvm.org/docs/\\1")

    description =
      Regex.replace(~r{\((.+?)\.md}, description, "(https://mlir.llvm.org/docs/Dialects/\\1")

    description =
      description
      |> String.replace("(Builtin/#", "(https://mlir.llvm.org/docs/Dialects/Builtin/#")

    summary = if summary != "", do: " - #{summary}", else: ""

    description =
      if description != "",
        do: """
        ## Description
        #{description}
        """,
        else: ""

    result_type_inference =
      if result_type_inference?(op) do
        "This op has support for result type inference."
      else
        ""
      end

    """
    `#{name}`#{summary}

    #{result_type_inference}
    """ <>
      gen_if_exist(op, "attributes") <>
      gen_if_exist(op, "operands") <>
      gen_if_exist(op, "results") <>
      description
  end

  def gen_doc(op) do
    case lookup(op) do
      {:ok, %{} = found} ->
        gen_doc(found)

      _ ->
        false
    end
  end

  def result_type_inference?(%{"result_type_inference" => result_type_inference}) do
    result_type_inference
  end

  def result_type_inference?(op) do
    case lookup(op) do
      {:ok, %{} = found} ->
        result_type_inference?(found)

      _ ->
        false
    end
  end
end
