ExUnit.configure(
  exclude: [
    stderr: true,
    cuda: :os.type() == {:unix, :darwin} or System.get_env("CI") == "true",
    cuda_runtime: :os.type() == {:unix, :darwin} or System.get_env("CI") == "true"
  ]
)

IO.puts("OS PID: #{System.pid()}")
ExUnit.start()
