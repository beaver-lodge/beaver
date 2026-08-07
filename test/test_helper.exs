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

ExUnit.configure(exclude: exclude)

IO.puts("OS PID: #{System.pid()}")
ExUnit.start()
