defmodule Beaver.BuildPlanTest do
  use ExUnit.Case, async: true
  alias Beaver.Build.{ArtifactCache, Plan}
  @tag :tmp_dir
  test "plans expose dependencies, cache verified results and stop on failed stages", %{tmp_dir: directory} do
    path = Path.join(directory, "analysis.json")
    checksum = path <> ".sha256"
    owner = self()

    build = fn ->
      send(owner, :analyzed)
      %{"answer" => 42}
    end

    plan =
      Plan.new()
      |> Plan.stage(:read, [], fn deps ->
        assert deps == %{}
        21
      end)
      |> Plan.stage(:analyze, [:read], fn %{read: 21} -> ArtifactCache.fetch_json!(path, checksum, build) end)
      |> Plan.stage(:publish, [:analyze], fn %{analyze: {value, hit?}} -> {value["answer"], hit?} end)

    assert Plan.describe(plan) == [
             %{name: :read, dependencies: []},
             %{name: :analyze, dependencies: [:read]},
             %{name: :publish, dependencies: [:analyze]}
           ]

    assert Plan.run!(plan).results.publish == {42, false}
    assert_received :analyzed
    assert Plan.run!(plan).results.publish == {42, true}
    refute_received :analyzed
    File.write!(path, "corrupt")
    assert_raise ArgumentError, ~r/cache digest mismatch/, fn -> Plan.run!(plan) end
    refute_received :analyzed
    assert_raise ArgumentError, fn -> Plan.stage(plan, :read, [], & &1) end
    assert_raise ArgumentError, fn -> Plan.stage(Plan.new(), :link, [:missing], & &1) end

    failed =
      Plan.new()
      |> Plan.stage(:compile, [], fn _ -> raise "failed compile" end)
      |> Plan.stage(:link, [:compile], fn _ -> send(owner, :linked) end)

    assert_raise RuntimeError, "failed compile", fn -> Plan.run!(failed) end
    refute_received :linked
  end

  @tag :tmp_dir
  test "incomplete cache entries rebuild and failed publication cleans its temporary file", %{tmp_dir: directory} do
    path = Path.join(directory, "artifact.json")
    checksum = path <> ".sha256"
    File.write!(checksum, "abandoned")
    assert {%{"valid" => true}, false} = ArtifactCache.fetch_json!(path, checksum, fn -> %{"valid" => true} end)
    destination = Path.join(directory, "published.json")
    ArtifactCache.copy_verified!(path, destination, File.read!(checksum))
    assert File.read!(path) == File.read!(destination)
    assert_raise ArgumentError, fn -> ArtifactCache.copy_verified!(path, destination, "wrong") end
    assert JSON.decode!(File.read!(destination)) == %{"valid" => true}
    assert_raise File.RenameError, fn -> ArtifactCache.write_atomic!(directory, "cannot replace directory") end
    assert Path.wildcard(directory <> ".tmp-*") == []
  end
end
