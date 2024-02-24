set -e
export MIX_ENV=prod
export ELIXIR_MAKE_CACHE_DIR=.
mix elixir_make.precompile
