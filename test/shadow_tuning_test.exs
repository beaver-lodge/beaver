defmodule Beaver.Shadow.TuningTest do
  use ExUnit.Case, async: true

  alias Beaver.Shadow.Tuning
  alias Beaver.Shadow.Tuning.{Config, Record, Run}

  @triton_enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil

  defp sample_configs do
    [
      %Config{index: 0, num_warps: 2},
      %Config{index: 1, num_warps: 4},
      %Config{index: 2, num_warps: 8, num_stages: 2}
    ]
  end

  defp sample_record(overrides \\ %{}) do
    base = %Record{
      format: Tuning.format(),
      kernel_digest: "a" <> String.duplicate("0", 63),
      target: "cuda:80",
      capability: nil,
      config_space_digest: Tuning.configs_digest(sample_configs()),
      config: %Config{index: 1, num_warps: 4},
      status: :evaluated,
      failure: nil,
      structural_proxy: %{baseline: 24, optimized: 5, reduction: 19, lowered_to_llvm: true},
      timings: %{total_ns: 1_000_000}
    }

    struct!(Record, Map.merge(Map.from_struct(base), overrides))
  end

  describe "config" do
    test "digest is stable and depends on the config fields" do
      a = %Config{index: 0, num_warps: 4}
      b = %Config{index: 0, num_warps: 4}
      c = %Config{index: 0, num_warps: 8}

      assert Config.digest(a) == Config.digest(b)
      refute Config.digest(a) == Config.digest(c)
    end
  end

  describe "record identity" do
    test "excludes durations and observations" do
      fast = sample_record(%{timings: %{total_ns: 1}})
      slow = sample_record(%{timings: %{total_ns: 99_999_999}})

      assert Tuning.identity(fast) == Tuning.identity(slow)
      refute fast.timings == slow.timings
    end

    test "includes the structural proxy and config provenance" do
      proxy_a =
        sample_record(%{
          structural_proxy: %{baseline: 24, optimized: 5, reduction: 19, lowered_to_llvm: true}
        })

      proxy_b =
        sample_record(%{
          structural_proxy: %{baseline: 24, optimized: 4, reduction: 20, lowered_to_llvm: true}
        })

      refute Tuning.identity(proxy_a) == Tuning.identity(proxy_b)
    end
  end

  describe "config space digest" do
    test "is index-independent" do
      configs = sample_configs()
      reindexed = Enum.map(configs, &%{&1 | index: 99})

      assert Tuning.configs_digest(configs) == Tuning.configs_digest(reindexed)
    end
  end

  describe "JSON round-trip" do
    test "encode!/decode! preserves the run" do
      run = %Run{
        kernel_digest: String.duplicate("b", 64),
        records: [sample_record()],
        winner: sample_record()
      }

      decoded = run |> Tuning.encode!() |> Tuning.decode!()

      assert decoded.kernel_digest == run.kernel_digest
      assert length(decoded.records) == 1
      assert decoded.records |> hd() |> Map.get(:config) |> Map.get(:num_warps) == 4
      assert decoded.winner.config.index == 1
      assert decoded.winner.structural_proxy.optimized == 5
    end
  end

  describe "AutotuneListener event" do
    test "carries the six upstream fields plus extensions" do
      run = %Run{
        kernel_digest: String.duplicate("c", 64),
        records: [
          sample_record(),
          sample_record(%{
            config: %Config{index: 2, num_warps: 8},
            status: :failed,
            structural_proxy: nil,
            failure: %{kind: :crash, reason: "segv"}
          }),
          sample_record(%{
            status: :failed,
            failure: %{kind: :lowering_failed, reason: "no llvm.func"}
          })
        ],
        winner: sample_record()
      }

      event = Tuning.event(run, fn_name: "matmul_kernel")

      # upstream protocol fields
      assert event.fn == "matmul_kernel"
      assert is_tuple(event.key)
      assert event.best_config.num_warps == 4
      assert is_list(event.configs_timings)
      assert is_integer(event.duration) or is_nil(event.duration)
      assert event.cache_hit == false

      # extensions
      assert is_map(event.structural_proxy)
      assert length(event.failure) == 2
    end
  end

  @tag :triton
  @tag skip: !@triton_enabled
  test "CPU-only config enumeration produces records and a structural winner" do
    run = Tuning.run(:matmul, sample_configs())

    assert run.kernel_digest |> byte_size() == 64
    assert length(run.records) == 3

    for record <- run.records do
      assert record.format == Tuning.format()
      assert record.status == :evaluated
      assert record.structural_proxy.baseline == 24
      assert is_boolean(record.structural_proxy.lowered_to_llvm)
      assert record.timings.total_ns > 0
    end

    assert run.winner.status == :evaluated
    assert run.winner.structural_proxy.optimized <= 5

    # re-running with the same configs reproduces identical decisions
    rerun = Tuning.run(:matmul, sample_configs())

    assert Enum.map(run.records, &Tuning.identity/1) ==
             Enum.map(rerun.records, &Tuning.identity/1)
  end
end
