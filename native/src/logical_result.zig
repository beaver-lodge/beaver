const prelude = @import("prelude.zig");

// MLIR implements these four functions as static inline C helpers. Export the
// same NIF names through Beaver's externally compiled C++ wrappers so Zig
// never owns the by-value MlirLogicalResult ABI boundary.
pub const nifs = .{
    prelude.nifAs("beaverLogicalResultSuccess", "mlirLogicalResultSuccess"),
    prelude.nifAs("beaverLogicalResultFailure", "mlirLogicalResultFailure"),
    prelude.nifAs("beaverLogicalResultIsSuccess", "mlirLogicalResultIsSuccess"),
    prelude.nifAs("beaverLogicalResultIsFailure", "mlirLogicalResultIsFailure"),
};
