set -e
export MIX_ENV=prod
export ELIXIR_MAKE_CACHE_DIR=.
llvm-config --version
mix deps.get
mix elixir_make.precompile
