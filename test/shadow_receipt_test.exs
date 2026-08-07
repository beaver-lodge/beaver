defmodule Beaver.Shadow.ReceiptTest do
  use ExUnit.Case, async: true

  alias Beaver.Shadow.Receipt

  defp receipt(overrides \\ %{}) do
    struct!(
      Receipt,
      Map.merge(
        %{
          format: Receipt.format(),
          source_digest: "abc",
          schedule: %{
            sequence: "__transform_main",
            digest: "schedule-digest",
            text: "module {}",
            bytecode: <<1, 2, 3>>
          },
          candidate: %{index: 0, choices: %{"tile" => 8}},
          artifact: %{cache: :miss, lookup_key: "lk", artifact_key: "ak"},
          trace: %{
            action_count: 3,
            tags: ["tile"],
            candidate_index: 0,
            schedule_digest: "schedule-digest"
          },
          status: :ok
        },
        overrides
      )
    )
  end

  test "identity is stable and excludes measurements" do
    assert Receipt.identity(receipt()) == Receipt.identity(receipt())

    assert Receipt.identity(receipt()) !=
             Receipt.identity(receipt(%{candidate: %{index: 1, choices: %{"tile" => 16}}}))
  end

  test "receipts round-trip through JSON" do
    original = receipt()
    decoded = original |> Receipt.encode!() |> Receipt.decode!()

    assert decoded.source_digest == "abc"
    assert decoded.schedule.bytecode == <<1, 2, 3>>
    assert decoded.artifact.cache == :miss
    assert decoded.status == :ok
    assert Receipt.identity(decoded) == Receipt.identity(original)
  end

  test "winner bytecode is retained for replay" do
    assert receipt().schedule.bytecode == <<1, 2, 3>>
    assert receipt().schedule.text == "module {}"
  end
end
