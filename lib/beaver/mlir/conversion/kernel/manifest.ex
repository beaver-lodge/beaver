defmodule Beaver.MLIR.Conversion.Kernel.Manifest do
  @moduledoc """
  Versioned, provider-neutral contract for an external conversion kernel.

  Binds an artifact to Beaver/LLVM revisions, dialect schema, runtime ABI,
  target, pattern roots and capabilities. Rejects unknown fields, duplicates,
  unsorted lists and identity mismatches before loading native code.
  Digests use JSON with sorted object keys and no insignificant whitespace;
  `identity_digest/1` excludes the artifact hash to avoid self-reference.
  """

  alias Beaver.MLIR.Conversion.Kernel.Error

  @schema_version 1
  @abi_version 1
  @digest_pattern ~r/^sha256:[0-9a-f]{64}$/
  @seed_kinds ~w(cpp-bootstrap previous-native native-kernel beam-reference)
  @stages ~w(stage0 stage1 stage2)
  @top_level_keys ~w(
    schema_version compiler_kernel_abi_version provider compiler_revision beaver_revision llvm_revision
    dialect_schema_digest runtime_abi_digest patterns capabilities target artifact_sha256 entrypoints bootstrap
  )
  @target_keys ~w(triple cpu features)
  @entrypoint_keys ~w(abi_version populate manifest)
  @bootstrap_keys ~w(stage seed provenance)
  @pattern_keys ~w(name root version)

  @enforce_keys Enum.map(@top_level_keys, &String.to_atom/1)
  defstruct @enforce_keys

  @type pattern() :: %{required(String.t()) => String.t()}
  @type target() :: %{required(String.t()) => String.t() | [String.t()]}
  @type entrypoints() :: %{required(String.t()) => String.t()}
  @type bootstrap() :: %{required(String.t()) => String.t()}
  @type t() :: %__MODULE__{
          schema_version: pos_integer(),
          compiler_kernel_abi_version: pos_integer(),
          provider: String.t(),
          compiler_revision: String.t(),
          beaver_revision: String.t(),
          llvm_revision: String.t(),
          dialect_schema_digest: String.t(),
          runtime_abi_digest: String.t(),
          patterns: [pattern()],
          capabilities: [String.t()],
          target: target(),
          artifact_sha256: String.t(),
          entrypoints: entrypoints(),
          bootstrap: bootstrap()
        }

  @doc "Returns the only manifest schema version understood by this Beaver build."
  @spec schema_version() :: pos_integer()
  def schema_version, do: @schema_version

  @doc "Returns the external compiler-kernel ABI version understood by this Beaver build."
  @spec abi_version() :: pos_integer()
  def abi_version, do: @abi_version

  @doc "Decodes and validates a JSON manifest."
  @spec decode!(iodata()) :: t()
  def decode!(json) do
    json |> IO.iodata_to_binary() |> JSON.decode!() |> new!()
  rescue
    error in JSON.DecodeError -> reject!(:invalid_manifest, %{reason: Exception.message(error)})
  end

  @doc "Builds and validates a manifest from a string-keyed map."
  @spec new!(map()) :: t()
  def new!(map) when is_map(map) do
    require_exact_keys!(map, @top_level_keys, "manifest")

    fields = Enum.map(@enforce_keys, &{&1, Map.fetch!(map, Atom.to_string(&1))})
    validate!(struct!(__MODULE__, fields))
  end

  def new!(other), do: reject!(:invalid_manifest, %{expected: "map", got: inspect(other)})

  @doc "Validates an already constructed manifest."
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{} = manifest) do
    require_version!(manifest.schema_version, @schema_version, :unsupported_schema)
    require_version!(manifest.compiler_kernel_abi_version, @abi_version, :abi_mismatch)

    for name <- ~w(provider compiler_revision beaver_revision llvm_revision)a,
        do: require_non_empty_string!(Map.fetch!(manifest, name), name)

    require_digest!(manifest.dialect_schema_digest, :dialect_schema_digest)
    require_digest_or_none!(manifest.runtime_abi_digest, :runtime_abi_digest)
    require_digest!(manifest.artifact_sha256, :artifact_sha256)
    validate_patterns!(manifest.patterns)
    require_sorted_unique_strings!(manifest.capabilities, :capabilities, non_empty: true)
    validate_target!(manifest.target)
    validate_entrypoints!(manifest.entrypoints)
    validate_bootstrap!(manifest.bootstrap)
    manifest
  end

  @doc "Returns the normalized string-keyed map used for transport and receipts."
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = manifest),
    do: Map.new(@enforce_keys, &{Atom.to_string(&1), Map.fetch!(manifest, &1)})

  @doc "Encodes the canonical JSON representation used by `digest/1`."
  @spec encode!(t()) :: binary()
  def encode!(%__MODULE__{} = manifest),
    do: manifest |> validate!() |> to_map() |> canonical_json()

  @doc "Returns the canonical manifest digest as a `sha256:` identity."
  @spec digest(t()) :: String.t()
  def digest(%__MODULE__{} = manifest),
    do: sha256(encode!(manifest))

  @doc """
  Returns the validated manifest digest excluding `artifact_sha256`.
  Embed this identity in the artifact for comparison by the loader.
  """
  @spec identity_digest(t()) :: String.t()
  def identity_digest(%__MODULE__{} = manifest) do
    canonical =
      manifest |> validate!() |> to_map() |> Map.delete("artifact_sha256") |> canonical_json()

    sha256(canonical)
  end

  @doc "Verifies the artifact bytes named by this manifest before loading them."
  @spec verify_artifact!(t(), Path.t()) :: t()
  def verify_artifact!(%__MODULE__{} = manifest, path) when is_binary(path) do
    actual =
      case File.read(path) do
        {:ok, bytes} -> sha256(bytes)
        {:error, reason} -> reject!(:artifact_unreadable, %{path: path, reason: reason})
      end

    require_identity!(actual, manifest.artifact_sha256, :artifact_digest_mismatch)
    manifest
  end

  @doc """
  Verifies runtime identities and required capabilities before loading a kernel.

  Only keys supplied in `expected` are checked. Supported keys are
  `:compiler_kernel_abi_version`, `:beaver_revision`, `:llvm_revision`,
  `:dialect_schema_digest`, `:runtime_abi_digest`, `:target`, and
  `:capabilities`.
  """
  @spec verify_compatible!(t(), keyword() | map()) :: t()
  def verify_compatible!(%__MODULE__{} = manifest, expected) do
    expected = expected_map!(expected)

    checks = [
      compiler_kernel_abi_version: :abi_mismatch,
      beaver_revision: :beaver_revision_mismatch,
      llvm_revision: :llvm_revision_mismatch,
      dialect_schema_digest: :dialect_schema_mismatch,
      runtime_abi_digest: :runtime_abi_mismatch,
      target: :target_mismatch
    ]

    case Map.keys(expected) -- [:capabilities | Keyword.keys(checks)] do
      [] -> :ok
      unknown -> reject!(:invalid_manifest, %{unknown_expectations: Enum.sort(unknown)})
    end

    for {key, code} <- checks,
        Map.has_key?(expected, key),
        do: require_identity!(Map.fetch!(manifest, key), expected[key], code)

    if required = expected[:capabilities] do
      require_sorted_unique_strings!(required, :capabilities, non_empty: false)
      missing = required -- manifest.capabilities
      if missing != [], do: reject!(:capability_missing, %{missing: missing})
    end

    manifest
  end

  defp validate_patterns!(patterns) when is_list(patterns) and patterns != [] do
    Enum.each(patterns, fn pattern ->
      require_exact_keys!(pattern, @pattern_keys, "pattern")
      Enum.each(@pattern_keys, &require_non_empty_string!(Map.fetch!(pattern, &1), &1))
    end)

    names = Enum.map(patterns, &Map.fetch!(&1, "name"))
    roots = Enum.map(patterns, &Map.fetch!(&1, "root"))
    require_sorted_unique_strings!(names, :pattern_names, non_empty: true)
    require_unique!(roots, :pattern_roots)
  end

  defp validate_patterns!(patterns),
    do: reject!(:invalid_manifest, %{field: :patterns, expected: "non-empty list", got: patterns})

  defp validate_target!(target) do
    require_exact_keys!(target, @target_keys, "target")
    require_non_empty_string!(Map.fetch!(target, "triple"), :target_triple)
    require_non_empty_string!(Map.fetch!(target, "cpu"), :target_cpu)
    require_sorted_unique_strings!(Map.fetch!(target, "features"), :target_features, non_empty: false)
  end

  defp validate_entrypoints!(entrypoints) do
    require_exact_keys!(entrypoints, @entrypoint_keys, "entrypoints")
    values = Enum.map(@entrypoint_keys, &Map.fetch!(entrypoints, &1))

    Enum.zip(@entrypoint_keys, values) |> Enum.each(fn {key, value} -> require_symbol!(value, key) end)

    require_unique!(values, :entrypoints)
  end

  defp validate_bootstrap!(bootstrap) do
    require_exact_keys!(bootstrap, @bootstrap_keys, "bootstrap")
    stage = Map.fetch!(bootstrap, "stage")
    seed = Map.fetch!(bootstrap, "seed")
    require_member!(stage, @stages, :bootstrap_stage)
    require_member!(seed, @seed_kinds, :bootstrap_seed)
    require_non_empty_string!(Map.fetch!(bootstrap, "provenance"), :bootstrap_provenance)
  end

  defp require_exact_keys!(map, expected, label) when is_map(map) do
    keys = Map.keys(map)

    unless Enum.all?(keys, &is_binary/1),
      do: reject!(:invalid_manifest, %{field: label, reason: "keys must be strings"})

    missing = expected -- keys
    unknown = keys -- expected

    if missing != [] or unknown != [],
      do: reject!(:invalid_manifest, %{field: label, missing: Enum.sort(missing), unknown: Enum.sort(unknown)})
  end

  defp require_exact_keys!(value, _expected, label),
    do: reject!(:invalid_manifest, %{field: label, expected: "map", got: value})

  defp require_version!(actual, expected, _code) when is_integer(actual) and actual == expected, do: :ok

  defp require_version!(actual, expected, code), do: reject!(code, %{expected: expected, actual: actual})

  defp require_non_empty_string!(value, _field) when is_binary(value) and byte_size(value) > 0, do: :ok

  defp require_non_empty_string!(value, field),
    do: reject!(:invalid_manifest, %{field: field, expected: "non-empty string", got: value})

  defp require_symbol!(value, field) do
    require_non_empty_string!(value, field)

    unless Regex.match?(~r/^[A-Za-z_][A-Za-z0-9_]*$/, value),
      do: reject!(:invalid_manifest, %{field: field, expected: "C symbol", got: value})
  end

  defp require_digest!(value, field) when is_binary(value),
    do: if(Regex.match?(@digest_pattern, value), do: :ok, else: reject_digest!(value, field))

  defp require_digest!(value, field),
    do: reject!(:invalid_manifest, %{field: field, expected: "sha256 digest", got: value})

  defp require_digest_or_none!("none", _field), do: :ok
  defp require_digest_or_none!(value, field), do: require_digest!(value, field)

  defp reject_digest!(value, field),
    do: reject!(:invalid_manifest, %{field: field, expected: "sha256:<64 lowercase hex>", got: value})

  defp require_member!(value, allowed, field),
    do: unless(value in allowed, do: reject!(:invalid_manifest, %{field: field, allowed: allowed, got: value}))

  defp require_sorted_unique_strings!(values, field, opts) when is_list(values) do
    unless Enum.all?(values, &(is_binary(&1) and byte_size(&1) > 0)),
      do: reject!(:invalid_manifest, %{field: field, expected: "list of non-empty strings"})

    if Keyword.fetch!(opts, :non_empty) and values == [],
      do: reject!(:invalid_manifest, %{field: field, expected: "non-empty list"})

    if values != Enum.sort(values) or length(values) != MapSet.size(MapSet.new(values)),
      do: reject!(:invalid_manifest, %{field: field, reason: "must be sorted and unique"})
  end

  defp require_sorted_unique_strings!(value, field, _opts),
    do: reject!(:invalid_manifest, %{field: field, expected: "list of strings", got: value})

  defp require_unique!(values, field) do
    if length(values) != MapSet.size(MapSet.new(values)),
      do: reject!(:invalid_manifest, %{field: field, reason: "must be unique"})
  end

  defp require_identity!(actual, expected, code),
    do: if(actual != expected, do: reject!(code, %{expected: expected, actual: actual}))

  defp expected_map!(expected) when is_list(expected) do
    if Keyword.keyword?(expected),
      do: Map.new(expected),
      else: reject!(:invalid_manifest, %{expected: "keyword or map"})
  end

  defp expected_map!(expected) when is_map(expected), do: expected
  defp expected_map!(_), do: reject!(:invalid_manifest, %{expected: "keyword or map"})

  defp canonical_json(value) when is_map(value) do
    value
    |> Map.to_list()
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map_join(",", fn {key, item} -> JSON.encode!(key) <> ":" <> canonical_json(item) end)
    |> then(&("{" <> &1 <> "}"))
  end

  defp canonical_json(value) when is_list(value),
    do: value |> Enum.map_join(",", &canonical_json/1) |> then(&("[" <> &1 <> "]"))

  defp canonical_json(value) when is_binary(value) or is_integer(value) or is_boolean(value) or is_nil(value),
    do: JSON.encode!(value)

  defp canonical_json(value),
    do: reject!(:invalid_manifest, %{reason: "value is not JSON-compatible", got: inspect(value)})

  defp sha256(bytes), do: "sha256:" <> Base.encode16(:crypto.hash(:sha256, bytes), case: :lower)

  defp reject!(code, details), do: raise(Error, code: code, details: details)
end
