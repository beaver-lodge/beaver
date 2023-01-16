import Config

import_config "#{config_env()}.exs"

if Mix.env() in [:dev, :test] do
  config :mix_test_watch,
    extra_extensions: [".zig", ".cpp"]
end
