exclude = [
  stderr: true,
  cuda: :os.type() != {:unix, :linux},
  cuda_runtime: :os.type() != {:unix, :linux} or System.get_env("CI") == "true"
]

exclude =
  if System.get_env("BEAVER_TRITON_PREBUILT_DIR") == nil do
    exclude ++ [:triton]
  else
    exclude
  end

exclude =
  if Beaver.MLIR.Transform.packed_params_supported?() do
    exclude
  else
    exclude ++ [:packed_transform_params]
  end

exclude =
  if Code.ensure_loaded?(Beaver.MLIR.Dialect.Transform) and
       function_exported?(
         Beaver.MLIR.Dialect.Transform,
         :apply_patterns_linalg_swap_extract_slice_with_fill,
         1
       ) do
    exclude
  else
    exclude ++ [:swap_extract_slice_with_fill]
  end

exclude =
  Enum.reduce(
    [
      {:optimization_remarks, Beaver.MLIR.OptimizationRemarks.supported?()},
      {:rocdl_target_info, Beaver.MLIR.Dialect.GPU.rocdl_target_info_supported?()},
      {:static_memory_planner,
       Beaver.MLIR.Dialect.Bufferization.static_memory_planner_supported?()}
    ],
    exclude,
    fn {tag, supported}, tags -> if supported, do: tags, else: [tag | tags] end
  )

ExUnit.configure(exclude: exclude)

IO.puts("OS PID: #{System.pid()}")
ExUnit.start()
