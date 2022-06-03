# $($LLVM_CONFIG_PATH --libdir)
set -eu
COMPILER_ARGS="-I scripts/zig/llvm-include -I $($LLVM_CONFIG_PATH --includedir)"
set -x
zig translate-c scripts/zig/llvm-include/14.h ${COMPILER_ARGS}
zig test scripts/zig/func-reflection.zig ${COMPILER_ARGS}
