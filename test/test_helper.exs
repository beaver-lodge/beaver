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
  if Beaver.MLIR.Transform.Schedule.DSL.alloc_to_global_supported?() and
       Beaver.MLIR.Transform.Schedule.DSL.loop_unroll_full_supported?() do
    exclude
  else
    exclude ++ [:recent_transform_helpers]
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

ExUnit.configure(exclude: exclude)

IO.puts("OS PID: #{System.pid()}")
ExUnit.start()
