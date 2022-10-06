// mlir-opt -pass-pipeline="func.func(tosa-to-linalg)" -cse --linalg-fuse-elementwise-ops -linalg-bufferize -tensor-bufferize -func-bufferize -buffer-results-to-out-params -convert-linalg-to-loops -convert-scf-to-cf -convert-linalg-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -reconcile-unrealized-casts test/tosa.mlir
module attributes {llvm.data_layout = ""} {
  func.func @test_multibroadcast(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> attributes {llvm.emit_c_interface} {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
