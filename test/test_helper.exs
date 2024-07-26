ExUnit.configure(exclude: [stderr: true, cuda: :os.type() == :darwin])
ExUnit.start()
