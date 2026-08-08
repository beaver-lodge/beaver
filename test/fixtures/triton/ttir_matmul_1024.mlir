module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> , %arg1: !tt.ptr<f32> , %arg2: !tt.ptr<f32> , %arg3: i32 , %arg4: i32 , %arg5: i32 , %arg6: i32 , %arg7: i32 , %arg8: i32 , %arg9: i32 , %arg10: i32 , %arg11: i32 ) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32> 
    %c0_i32 = arith.constant 0 : i32 
    %c64_i32 = arith.constant 64 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = tt.get_program_id y : i32 
    %2 = arith.muli %0, %c64_i32 : i32 
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
    %4 = tt.splat %2 : i32 -> tensor<64xi32> 
    %5 = arith.addi %4, %3 : tensor<64xi32> 
    %6 = arith.muli %1, %c64_i32 : i32 
    %7 = tt.splat %6 : i32 -> tensor<64xi32> 
    %8 = arith.addi %7, %3 : tensor<64xi32> 
    %9 = tt.expand_dims %5 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32> 
    %10 = tt.splat %arg6 : i32 -> tensor<64x1xi32> 
    %11 = arith.muli %9, %10 : tensor<64x1xi32> 
    %12 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>> 
    %13 = tt.addptr %12, %11 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32> 
    %14 = tt.expand_dims %3 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32> 
    %15 = tt.splat %arg7 : i32 -> tensor<1x64xi32> 
    %16 = arith.muli %14, %15 : tensor<1x64xi32> 
    %17 = tt.broadcast %13 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>> 
    %18 = tt.broadcast %16 : tensor<1x64xi32> -> tensor<64x64xi32> 
    %19 = tt.addptr %17, %18 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32> 
    %20 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32> 
    %21 = tt.splat %arg8 : i32 -> tensor<64x1xi32> 
    %22 = arith.muli %20, %21 : tensor<64x1xi32> 
    %23 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>> 
    %24 = tt.addptr %23, %22 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32> 
    %25 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32> 
    %26 = tt.splat %arg9 : i32 -> tensor<1x64xi32> 
    %27 = arith.muli %25, %26 : tensor<1x64xi32> 
    %28 = tt.broadcast %24 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>> 
    %29 = tt.broadcast %27 : tensor<1x64xi32> -> tensor<64x64xi32> 
    %30 = tt.addptr %28, %29 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32> 
    %31:3 = scf.for %arg12 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg13 = %cst, %arg14 = %19, %arg15 = %30) -> (tensor<64x64xf32>, tensor<64x64x!tt.ptr<f32>>, tensor<64x64x!tt.ptr<f32>>)  : i32 {
      %41 = tt.load %arg14 : tensor<64x64x!tt.ptr<f32>> 
      %42 = tt.load %arg15 : tensor<64x64x!tt.ptr<f32>> 
      %43 = tt.dot %41, %42, %arg13, inputPrecision = tf32 : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32> 
      %44 = arith.muli %arg7, %c64_i32 : i32 
      %45 = tt.splat %44 : i32 -> tensor<64x64xi32> 
      %46 = tt.addptr %arg14, %45 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32> 
      %47 = arith.muli %arg8, %c64_i32 : i32 
      %48 = tt.splat %47 : i32 -> tensor<64x64xi32> 
      %49 = tt.addptr %arg15, %48 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32> 
      scf.yield %43, %46, %49 : tensor<64x64xf32>, tensor<64x64x!tt.ptr<f32>>, tensor<64x64x!tt.ptr<f32>> 
    } 
    %32 = tt.splat %arg10 : i32 -> tensor<64x1xi32> 
    %33 = arith.muli %9, %32 : tensor<64x1xi32> 
    %34 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>> 
    %35 = tt.addptr %34, %33 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32> 
    %36 = tt.splat %arg11 : i32 -> tensor<1x64xi32> 
    %37 = arith.muli %25, %36 : tensor<1x64xi32> 
    %38 = tt.broadcast %35 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>> 
    %39 = tt.broadcast %37 : tensor<1x64xi32> -> tensor<64x64xi32> 
    %40 = tt.addptr %38, %39 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32> 
    tt.store %40, %31#0 : tensor<64x64x!tt.ptr<f32>> 
    tt.return 
  } 
} 