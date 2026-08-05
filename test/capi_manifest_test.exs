defmodule Beaver.MLIR.CAPI.ManifestTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR.CAPI
  alias Kinda.CodeGen.DeclarationManifest

  @moduletag :smoke

  test "declaration manifest matches the generated public and raw surfaces" do
    declaration_manifest = CAPI.CodeGen.declaration_manifest()
    declarations = DeclarationManifest.nif_decls(declaration_manifest)

    expected =
      declarations
      |> Enum.map(fn declaration ->
        arity =
          if is_list(declaration.params), do: length(declaration.params), else: declaration.params

        {declaration.wrapper_name, arity}
      end)
      |> MapSet.new()

    raw = CAPI.Raw.__info__(:functions) |> MapSet.new()
    public = CAPI.__info__(:functions) |> MapSet.new()

    assert MapSet.size(expected) > 2_000
    assert MapSet.subset?(expected, raw)
    assert MapSet.subset?(expected, public)
  end

  test "declaration manifest preserves signature and wrapper policy metadata" do
    declaration_manifest = CAPI.CodeGen.declaration_manifest()
    declarations = DeclarationManifest.nif_decls(declaration_manifest)

    assert Enum.any?(declarations, &(is_binary(&1.doc) and &1.doc != ""))

    assert Enum.any?(
             declarations,
             &Enum.any?(&1.param_ctypes || [], fn ctype -> ctype != nil end)
           )

    assert Enum.any?(declarations, &(&1.return_ctype != nil))

    assert %{dirty: :dirty_cpu} =
             Enum.find(
               declarations,
               &(&1.wrapper_name == :mlirExecutionEngineInvokePacked_dirty_cpu)
             )

    assert %{dirty: :dirty_io} =
             Enum.find(
               declarations,
               &(&1.wrapper_name == :mlirExecutionEngineInvokePacked_dirty_io)
             )

    assert %{params: [:context | _], dirty: false} =
             Enum.find(declarations, &(&1.wrapper_name == :mlirOperationVerifyWithDiagnostics))
  end

  test "callback-heavy declarations remain in the callback bridge manifest" do
    priv_dir = :beaver |> :code.priv_dir() |> List.to_string()

    callback_entries =
      priv_dir
      |> Path.join("capi_callback_bridge.json")
      |> File.read!()
      |> Jason.decode!()
      |> Map.fetch!("entries")

    declaration_manifest = CAPI.CodeGen.declaration_manifest()

    emitted_names =
      declaration_manifest
      |> DeclarationManifest.nif_decls()
      |> Enum.map(&Atom.to_string(&1.wrapper_name))
      |> MapSet.new()

    assert callback_entries != []

    assert Enum.any?(
             callback_entries,
             &(get_in(&1, ["function", "name"]) == "mlirValueReplaceUsesWithIf")
           )

    for entry <- callback_entries do
      name = get_in(entry, ["function", "name"])
      assert get_in(entry, ["callback_bridge", "reason"]) == "callback_bridge_required"
      refute MapSet.member?(emitted_names, name)
    end
  end

  test "build does not install generated wrapper sources" do
    priv_dir = :beaver |> :code.priv_dir() |> List.to_string()

    refute File.exists?(Path.join(priv_dir, "capi_functions.ex"))
    refute File.exists?(Path.join(priv_dir, "generated/wrapper.zig"))
  end
end
