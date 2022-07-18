# run this script to run `mix test` under lldb
# erlang's executable is not directly run so we dry-run an elixir command to get the full command and then run it under lldb

set -e
ROOT_DIR=$(elixir --eval ":code.root_dir() |> IO.puts()")
VERSION=$(elixir --eval ":erlang.system_info(:version) |> IO.puts()")
export BINDIR=$ROOT_DIR/erts-$VERSION/bin
EXE=$BINDIR/erlexec
echo "ROOT_DIR: $ROOT_DIR"
echo "VERSION: $VERSION"
echo "EXE: $EXE"
FULL=$(ELIXIR_CLI_DRY_RUN=1 mix test)
FULL=${FULL/erl/${EXE}}
echo "FULL: $FULL"
lldb -- $FULL
