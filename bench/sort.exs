Beaver.MIF.init_jit(ENIFQuickSort)
Beaver.MIF.init_jit(ENIFMergeSort)
Beaver.MIF.init_jit(ENIFTimSort)

Benchee.run(
  %{
    "Enum.sort" => fn arr -> Enum.sort(arr) end,
    "enif_quick_sort" => fn arr -> ENIFQuickSort.sort(arr, :arg_err) end,
    "enif_merge_sort" => fn arr -> ENIFMergeSort.sort(arr, :arg_err) end,
    "enif_tim_sort" => fn arr -> ENIFTimSort.sort(arr, :arg_err) end
  },
  time: 10,
  before_scenario: fn _ ->
    Enum.to_list(1..67_000) |> Enum.shuffle()
  end
)

Beaver.MIF.destroy_jit(ENIFMergeSort)
Beaver.MIF.destroy_jit(ENIFQuickSort)
Beaver.MIF.destroy_jit(ENIFTimSort)
