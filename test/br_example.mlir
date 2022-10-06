// To debug, run mlir-opt --mlir-print-op-generic test/br_example.mlir

// CHECK-LABEL: func @up_propagate
func.func @up_propagate() -> i32 {
  // CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
  %0 = arith.constant 0 : i32

  // CHECK-NEXT: %true = arith.constant true
  %cond = arith.constant true

  // CHECK-NEXT: cf.cond_br %true, ^bb1, ^bb2(%c0_i32 : i32)
  cf.cond_br %cond, ^bb1, ^bb2(%0 : i32)

^bb1: // CHECK: ^bb1:
  // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32

  // CHECK-NEXT: cf.br ^bb2(%c1_i32 : i32)
  cf.br ^bb2(%1 : i32)

^bb2(%arg : i32): // CHECK: ^bb2
  // CHECK-NEXT: %c1_i32_0 = arith.constant 1 : i32
  %2 = arith.constant 1 : i32

  // CHECK-NEXT: %1 = arith.addi %0, %c1_i32_0 : i32
  %add = arith.addi %arg, %2 : i32

  // CHECK-NEXT: return %1 : i32
  return %add : i32
}
