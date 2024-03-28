Beaver.MIF.init_jit(ENIFQuickSort)
Beaver.MIF.init_jit(ENIFMergeSort)

arr = Enum.to_list(1..60_000) |> Enum.shuffle()

Benchee.run(%{
  "Enum.sort" => fn -> Enum.sort(arr) end,
  "enif_quick_sort" => fn -> ENIFQuickSort.sort(arr, :arg_err) end,
  "enif_merge_sort" => fn -> ENIFMergeSort.sort(arr, :arg_err) end
})

Beaver.MIF.destroy_jit(ENIFMergeSort)
Beaver.MIF.destroy_jit(ENIFQuickSort)
