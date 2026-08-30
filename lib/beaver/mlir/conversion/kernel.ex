defmodule Beaver.MLIR.Conversion.Kernel do
  @moduledoc """
  Provider-neutral loading boundary for external native conversion kernels.

  Loading verifies the sidecar manifest and artifact bytes in Elixir before a
  Beaver-owned native loader resolves any artifact entrypoint. The native
  boundary then verifies the exported ABI version, embedded manifest identity,
  and exact pattern population. Artifacts are content addressed and remain
  loaded for the lifetime of the VM so MLIR pattern function pointers cannot
  become stale.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Kernel.{Error, Manifest}

  @native_error_codes %{
    "E_INVALID_ARGUMENT" => :invalid_loader_argument,
    "E_DLOPEN" => :loader_error,
    "E_MISSING_SYMBOL" => :missing_symbol,
    "E_ABI_MISMATCH" => :abi_mismatch,
    "E_MANIFEST_IDENTITY_MISMATCH" => :manifest_identity_mismatch,
    "E_PATTERN_MANIFEST_INVALID" => :pattern_manifest_invalid,
    "E_PATTERN_MANIFEST_MISMATCH" => :pattern_manifest_mismatch,
    "E_POPULATE_FAILED" => :population_failed,
    "E_EXCEPTION" => :native_exception
  }

  @doc "Validates that an artifact path is explicit and independent of cwd."
  @spec validate_artifact_path!(Path.t()) :: Path.t()
  def validate_artifact_path!(path) when is_binary(path) do
    if Path.type(path) == :absolute do
      path
    else
      raise Error,
        code: :invalid_artifact_path,
        details: %{path: path, reason: "compiler-kernel artifact path must be absolute"}
    end
  end

  def validate_artifact_path!(path) do
    raise Error,
      code: :invalid_artifact_path,
      details: %{path: inspect(path), reason: "compiler-kernel artifact path must be a string"}
  end

  @doc """
  Verifies, loads, and populates patterns from an external compiler kernel.

  `:expected` accepts the identity checks supported by
  `Manifest.verify_compatible!/2`. The linked compiler-kernel ABI and LLVM
  revision are always checked and cannot be relaxed by the caller.
  """
  @spec populate!(
          MLIR.RewritePatternSet.t(),
          MLIR.TypeConverter.t(),
          Manifest.t(),
          Path.t(),
          keyword()
        ) :: :ok
  def populate!(patterns, converter, %Manifest{} = manifest, artifact_path, opts \\ []) do
    unless match?(%MLIR.RewritePatternSet{}, patterns) and
             match?(%MLIR.TypeConverter{}, converter) do
      raise ArgumentError,
            "external kernel population requires pattern-set and type-converter handles"
    end

    unless Keyword.keyword?(opts) and Keyword.keys(opts) -- [:expected] == [] do
      raise ArgumentError, "unsupported compiler-kernel population options: #{inspect(opts)}"
    end

    artifact_path = validate_artifact_path!(artifact_path)

    expected =
      opts
      |> Keyword.get(:expected, [])
      |> normalize_expected!()
      |> Keyword.put(:compiler_kernel_abi_version, Manifest.abi_version())
      |> Keyword.put(:llvm_revision, MLIR.CompilationRuntime.llvm_revision())

    manifest
    |> Manifest.validate!()
    |> Manifest.verify_compatible!(expected)
    |> Manifest.verify_artifact!(artifact_path)

    entrypoints = manifest.entrypoints

    error =
      MLIR.CAPI.beaver_raw_compiler_kernel_load_and_populate(
        patterns.ref,
        converter.ref,
        artifact_path,
        entrypoints["abi_version"],
        entrypoints["manifest"],
        entrypoints["populate"],
        Manifest.identity_digest(manifest),
        JSON.encode!(manifest.patterns)
      )

    case error do
      "" -> :ok
      native_error -> raise_native_error!(native_error, artifact_path)
    end
  end

  defp normalize_expected!(expected) when is_list(expected) do
    if Keyword.keyword?(expected) do
      expected
    else
      raise ArgumentError, ":expected must be a keyword list"
    end
  end

  defp normalize_expected!(expected) when is_map(expected), do: Map.to_list(expected)

  defp normalize_expected!(_expected),
    do: raise(ArgumentError, ":expected must be a keyword or map")

  defp raise_native_error!(native_error, artifact_path) do
    {native_code, message} =
      case String.split(native_error, "|", parts: 2) do
        [code, message] -> {code, message}
        [message] -> {"E_EXCEPTION", message}
      end

    raise Error,
      code: Map.get(@native_error_codes, native_code, :native_exception),
      message: message,
      details: %{artifact: artifact_path, native_code: native_code}
  end
end
