defmodule Beaver.MLIR.ODS.Dump do
  @moduledoc """
  This module provides functionality for working with MLIR ODS (Operation Definition Specification) dumps.
  It allows looking up operation definitions by name and generating documentation for them.

  The module loads ODS dump data at compile time and provides functions to:
  - Look up operations by their fully qualified names (e.g. "affine.for")
  - Generate documentation for operations including their attributes, operands, and results
  - Check if an operation supports result type inference

  Operations can be looked up using `lookup/1`, and documentation can be generated using `gen_doc/1`.
  """
  @dump Application.app_dir(:beaver)
        |> Path.join("priv/ods_dump.ex")
        |> File.read!()
        |> Code.eval_string()
        |> elem(0)

  @dialects @dump
            |> Enum.flat_map(fn {"dialects", dialects} ->
              dialects
            end)
  @doc """
  Lookup an operation by name (e.g. "affine.for").
  """
  for %{"operations" => operations} <- @dialects do
    for %{"name" => name} = op <- operations do
      def lookup(unquote(name)) do
        {:ok, unquote(Macro.escape(op))}
      end
    end
  end

  def lookup(op) do
    {:error, "fail to found ods dump of #{inspect(op)}"}
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
    summary = if summary != "" and summary != nil, do: " - #{summary}", else: ""

    result_type_inference =
      if result_type_inference?(op) do
        "This op has support for result type inference."
      else
        ""
      end

    """
    `#{name}`#{summary}

    #{op["description"]}

    #{result_type_inference}
    """ <>
      gen_if_exist(op, "attributes") <>
      gen_if_exist(op, "operands") <> gen_if_exist(op, "results")
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
