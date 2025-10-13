set -e
export MIX_ENV=prod
export ELIXIR_MAKE_CACHE_DIR=.

${LLVM_CONFIG_PATH:-llvm-config} --version
mix deps.get
mix elixir_make.precompile
