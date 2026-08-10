defmodule Beaver.MLIR.Dialect.DLTI do
  @moduledoc """
  Operations and structured attribute builders for MLIR DLTI.

  DLTI attributes are contextual, so all builders follow Beaver's deferred
  convention: pass `ctx:` for eager creation or pass the returned deferred value to
  another builder or operation.
  """

  alias Beaver.MLIR

  use Beaver.MLIR.Dialect,
    dialect: "dlti",
    ops: Beaver.MLIR.Dialect.Registry.ops("dlti")

  @data_layout_attribute "dlti.dl_spec"
  @target_system_attribute "dlti.target_system_spec"

  def data_layout_attribute_name, do: @data_layout_attribute
  def target_system_attribute_name, do: @target_system_attribute

  @doc """
  Build one DLTI entry from a string/type key and an attribute-compatible value.

  Strings, atoms, integers, booleans, MLIR attributes and deferred attributes
  are accepted as values. Types and deferred types are accepted as keys.
  """
  def entry(key, value, opts \\ []) do
    Beaver.Deferred.from_opts(opts, fn ctx ->
      MLIR.Attribute.get(entry_text(key, value, ctx), ctx: ctx)
    end)
  end

  @doc """
  Build a `#dlti.dl_spec` from `{key, value}` pairs or existing DLTI entries.
  """
  def spec(entries, opts \\ []) when is_list(entries) do
    build_entry_container("dl_spec", entries, opts)
  end

  @doc """
  Build a common data-layout spec.

  Supported semantic options are `:endianness` (`:little` or `:big`) and
  `:mangling_mode`. Additional `{key, value}` pairs may be supplied through
  `:entries`.
  """
  def data_layout(opts \\ []) when is_list(opts) do
    entries =
      []
      |> maybe_add(:endianness, opts, fn value ->
        unless value in [:little, :big, "little", "big"] do
          raise ArgumentError, "endianness must be :little or :big"
        end

        {"dlti.endianness", value}
      end)
      |> maybe_add(:mangling_mode, opts, &{"dlti.mangling_mode", &1})
      |> Kernel.++(Keyword.get(opts, :entries, []))

    spec(entries, Keyword.take(opts, [:ctx]))
  end

  @doc "Build a `#dlti.target_device_spec` from structured entries."
  def target_device_spec(entries, opts \\ []) when is_list(entries) do
    build_entry_container("target_device_spec", entries, opts)
  end

  @doc """
  Build a target-system spec from `{device_id, entries_or_device_spec}` pairs.
  """
  def target_system_spec(devices, opts \\ []) when is_list(devices) do
    Beaver.Deferred.from_opts(opts, fn ctx ->
      entries =
        Enum.map(devices, fn
          {device_id, entries} when is_list(entries) ->
            {device_id, target_device_spec(entries, ctx: ctx)}

          {device_id, device_spec} ->
            {device_id, device_spec}

          other ->
            raise ArgumentError,
                  "target devices must be {device_id, entries_or_spec} pairs, got: #{inspect(other)}"
        end)

      text = container_text("target_system_spec", entries, ctx)
      MLIR.Attribute.get(text, ctx: ctx)
    end)
  end

  @doc "Attach a data-layout spec to an MLIR module or operation."
  def attach(operation_or_module, spec) do
    put_attribute(operation_or_module, @data_layout_attribute, spec)
  end

  @doc "Attach a target-system spec to an MLIR module or operation."
  def attach_target_system(operation_or_module, target_system_spec) do
    put_attribute(operation_or_module, @target_system_attribute, target_system_spec)
  end

  defp put_attribute(operation_or_module, name, attribute) do
    operation = MLIR.Operation.from_module(operation_or_module)
    attribute = Beaver.Deferred.resolve(attribute, MLIR.context(operation))
    MLIR.Operation.get_and_update(operation, name, fn _ -> {nil, attribute} end)
    operation_or_module
  end

  defp build_entry_container(name, entries, opts) do
    Beaver.Deferred.from_opts(opts, fn ctx ->
      MLIR.Attribute.get(container_text(name, entries, ctx), ctx: ctx)
    end)
  end

  defp container_text(name, entries, ctx) do
    rendered = Enum.map(entries, &render_entry(&1, ctx))
    reject_duplicate_entries!(rendered)
    "#dlti.#{name}<#{Enum.join(rendered, ", ")}>"
  end

  defp render_entry({key, value}, ctx), do: entry_text(key, value, ctx)

  defp render_entry(entry, ctx) do
    entry
    |> Beaver.Deferred.resolve(ctx)
    |> to_string()
  end

  defp entry_text(key, value, ctx) do
    "#dlti.dl_entry<#{render_key(key, ctx)}, #{render_value(value, ctx)}>"
  end

  defp render_key(key, _ctx) when is_binary(key), do: JSON.encode!(key)
  defp render_key(key, ctx) when is_atom(key), do: render_key(Atom.to_string(key), ctx)

  defp render_key(key, ctx) do
    key
    |> Beaver.Deferred.resolve(ctx)
    |> case do
      %MLIR.Type{} = type ->
        to_string(type)

      other ->
        raise ArgumentError,
              "DLTI entry key must be a string, atom, or MLIR type, got: #{inspect(other)}"
    end
  end

  defp render_value(value, ctx) when is_binary(value),
    do: value |> MLIR.Attribute.string(ctx: ctx) |> to_string()

  defp render_value(value, ctx) when is_integer(value),
    do: MLIR.Attribute.integer(MLIR.Type.i64(ctx: ctx), value) |> to_string()

  defp render_value(value, ctx) when is_boolean(value),
    do: MLIR.Attribute.bool(value, ctx: ctx) |> to_string()

  defp render_value(value, ctx) when is_atom(value), do: render_value(Atom.to_string(value), ctx)

  defp render_value(value, ctx) do
    value
    |> Beaver.Deferred.resolve(ctx)
    |> case do
      %MLIR.Attribute{} = attribute -> to_string(attribute)
      %MLIR.Type{} = type -> type |> MLIR.Attribute.type() |> to_string()
      other -> raise ArgumentError, "unsupported DLTI entry value: #{inspect(other)}"
    end
  end

  defp reject_duplicate_entries!(entries) do
    keys = Enum.map(entries, &entry_key!/1)

    case keys -- Enum.uniq(keys) do
      [] -> :ok
      [duplicate | _] -> raise ArgumentError, "duplicate DLTI entry key: #{duplicate}"
    end
  end

  defp entry_key!(entry) do
    case Regex.run(~r/^#dlti\.dl_entry<(.+?), /, entry, capture: :all_but_first) do
      [key] -> key
      _ -> entry
    end
  end

  defp maybe_add(entries, key, opts, fun) do
    case Keyword.fetch(opts, key) do
      {:ok, value} -> entries ++ [fun.(value)]
      :error -> entries
    end
  end
end
