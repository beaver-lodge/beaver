defmodule Beaver.MLIR.Conversion.Ex.Stage0 do
  @moduledoc """
  Machine-readable boundary of Beaver's frozen C++ Ex bootstrap seed.

  This seed exists only to compile Batata's restricted compiler-kernel source
  from a clean checkout and to serve as a differential oracle. It is not a
  production provider and must not grow with Batata runtime or standard-library
  semantics. Evolving production patterns belong to Batata.

  The identity digest covers the schema version, native entrypoint, and every
  ordered pattern declaration. Any seed change therefore requires an explicit
  manifest and bootstrap receipt update.
  """

  @schema_version 1
  @entrypoint "beaverPopulateExScalarConversionPatterns"
  @patterns [
    %{"name" => "cpp.ex.add.v1", "root" => "ex.add", "version" => "1.0"},
    %{"name" => "cpp.ex.box.v1", "root" => "ex.box", "version" => "1.0"},
    %{"name" => "cpp.ex.cmp.v1", "root" => "ex.cmp", "version" => "1.0"},
    %{"name" => "cpp.ex.div.v1", "root" => "ex.div", "version" => "1.0"},
    %{"name" => "cpp.ex.if.v1", "root" => "ex.if", "version" => "1.0"},
    %{"name" => "cpp.ex.lit.v1", "root" => "ex.lit", "version" => "1.0"},
    %{"name" => "cpp.ex.mul.v1", "root" => "ex.mul", "version" => "1.0"},
    %{"name" => "cpp.ex.rem.v1", "root" => "ex.rem", "version" => "1.0"},
    %{"name" => "cpp.ex.sub.v1", "root" => "ex.sub", "version" => "1.0"},
    %{"name" => "cpp.ex.to_word.v1", "root" => "ex.to_word", "version" => "1.0"},
    %{"name" => "cpp.ex.unbox.v1", "root" => "ex.unbox", "version" => "1.0"},
    %{"name" => "cpp.ex.yield.v1", "root" => "ex.yield", "version" => "1.0"}
  ]

  @identity_payload [
                      Integer.to_string(@schema_version),
                      @entrypoint,
                      Enum.map_join(@patterns, "\n", fn pattern ->
                        Enum.join(
                          [pattern["name"], pattern["root"], pattern["version"]],
                          "\0"
                        )
                      end)
                    ]
                    |> Enum.join("\n")

  @identity_digest "sha256:" <>
                     Base.encode16(:crypto.hash(:sha256, @identity_payload), case: :lower)

  @manifest %{
    "schema_version" => @schema_version,
    "provider" => "cpp-bootstrap",
    "entrypoint" => @entrypoint,
    "patterns" => @patterns,
    "identity_digest" => @identity_digest
  }

  @doc "Returns the frozen bootstrap-seed manifest."
  @spec manifest() :: map()
  def manifest, do: @manifest

  @doc "Returns the exact Ex roots implemented by the frozen C++ seed."
  @spec roots() :: [String.t()]
  def roots, do: Enum.map(@patterns, & &1["root"])

  @doc "Returns the content identity used by bootstrap receipts."
  @spec identity_digest() :: String.t()
  def identity_digest, do: @identity_digest
end
