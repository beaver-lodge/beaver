// To debug, run mlir-opt --mlir-print-op-generic apps/mlir/test/pdl_erase_and_create.mlir
pdl.pattern : benefit(1) {
  %root = pdl.operation "test.op"
  pdl.rewrite %root {
    pdl.operation "test.success2"
    pdl.erase %root
  }
}
