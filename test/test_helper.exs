ExUnit.configure(
  exclude: [
    stderr: true,
    cuda: :os.type() == {:unix, :darwin},
    cuda_runtime: :os.type() == {:unix, :darwin} or System.get_env("CI") == "true"
  ],
  timeout: 10_000
)

ExUnit.start()
