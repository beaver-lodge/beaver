// Generated from the flash-attention-style forward kernel in
// triton-lang/triton python frontend 3.4.0
// (N_CTX=64, BLOCK_M=BLOCK_N=64, HEAD_DIM=64, num_warps=4, target sm_80).
// The kernel keeps online-softmax accumulators (m_i/l_i) as scf.for
// iter_args and uses tt.dot for the QK^T and PV matmuls, so lowering
// produces a realistic mix of layout conversions inside the loop.
//
// RUN: triton-opt %s -convert-triton-to-tritongpu=target=cuda:80 -tritongpu-remove-layout-conversions -canonicalize | FileCheck %s
module {
  tt.func public @attn_fwd(%arg0: !tt.ptr<f32> , %arg1: !tt.ptr<f32> , %arg2: !tt.ptr<f32> , %arg3: f32 , %arg4: !tt.ptr<f32> , %arg5: i32 , %arg6: i32 , %arg7: i32 , %arg8: i32 , %arg9: i32 , %arg10: i32 , %arg11: i32 , %arg12: i32 , %arg13: i32 , %arg14: i32 , %arg15: i32 , %arg16: i32 , %arg17: i32 , %arg18: i32 , %arg19: i32 , %arg20: i32 ) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x1xf32> 
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32> 
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<64xf32> 
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64xf32> 
    %cst_3 = arith.constant dense<0xFF800000> : tensor<64xf32> 
    %c64_i32 = arith.constant 64 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c64_i32 : i32 
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
    %3 = tt.splat %1 : i32 -> tensor<64xi32> 
    %4 = arith.addi %3, %2 : tensor<64xi32> 
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32> 
    %6 = tt.splat %arg7 : i32 -> tensor<64x1xi32> 
    %7 = arith.muli %5, %6 : tensor<64x1xi32> 
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>> 
    %9 = tt.addptr %8, %7 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32> 
    %10 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32> 
    %11 = tt.splat %arg8 : i32 -> tensor<1x64xi32> 
    %12 = arith.muli %10, %11 : tensor<1x64xi32> 
    %13 = tt.broadcast %9 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>> 
    %14 = tt.broadcast %12 : tensor<1x64xi32> -> tensor<64x64xi32> 
    %15 = tt.addptr %13, %14 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32> 
    %16 = tt.load %15 : tensor<64x64x!tt.ptr<f32>> 
    %17 = tt.splat %arg11 : i32 -> tensor<1x64xi32> 
    %18 = arith.muli %10, %17 : tensor<1x64xi32> 
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>> 
    %20 = tt.addptr %19, %18 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32> 
    %21 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32> 
    %22 = tt.splat %arg12 : i32 -> tensor<64x1xi32> 
    %23 = arith.muli %21, %22 : tensor<64x1xi32> 
    %24 = tt.broadcast %20 : tensor<1x64x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>> 
    %25 = tt.broadcast %23 : tensor<64x1xi32> -> tensor<64x64xi32> 
    %26 = tt.addptr %24, %25 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32> 
    %27 = tt.splat %arg15 : i32 -> tensor<64x1xi32> 
    %28 = arith.muli %21, %27 : tensor<64x1xi32> 
    %29 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>> 
    %30 = tt.addptr %29, %28 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32> 
    %31 = tt.splat %arg16 : i32 -> tensor<1x64xi32> 
    %32 = arith.muli %10, %31 : tensor<1x64xi32> 
    %33 = tt.broadcast %30 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>> 
    %34 = tt.broadcast %32 : tensor<1x64xi32> -> tensor<64x64xi32> 
    %35 = tt.addptr %33, %34 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32> 
    %36 = tt.load %26 : tensor<64x64x!tt.ptr<f32>> 
    %37 = tt.dot %16, %36, %cst_0, inputPrecision = tf32 : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32> 
    %38 = tt.splat %arg3 : f32 -> tensor<64x64xf32> 
    %39 = arith.mulf %37, %38 : tensor<64x64xf32> 
    %40 = "tt.reduce"(%39) <{axis = 1 : i32}> ({
    ^bb0(%arg21: f32 , %arg22: f32 ):
      %72 = arith.maxnumf %arg21, %arg22 : f32 
      tt.reduce.return %72 : f32 
    }) : (tensor<64x64xf32>) -> tensor<64xf32> 
    %41 = arith.maxnumf %40, %cst_3 : tensor<64xf32> 
    %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> 
    %43 = tt.broadcast %42 : tensor<64x1xf32> -> tensor<64x64xf32> 
    %44 = arith.subf %39, %43 : tensor<64x64xf32> 
    %45 = math.exp %44 : tensor<64x64xf32> 
    %46 = "tt.reduce"(%45) <{axis = 1 : i32}> ({
    ^bb0(%arg21: f32 , %arg22: f32 ):
      %72 = arith.addf %arg21, %arg22 : f32 
      tt.reduce.return %72 : f32 
    }) : (tensor<64x64xf32>) -> tensor<64xf32> 
    %47 = arith.subf %cst_3, %41 : tensor<64xf32> 
    %48 = math.exp %47 : tensor<64xf32> 
    %49 = arith.mulf %48, %cst_2 : tensor<64xf32> 
    %50 = arith.addf %49, %46 : tensor<64xf32> 
    %51 = tt.expand_dims %48 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> 
    %52 = arith.mulf %51, %cst : tensor<64x1xf32> 
    %53 = tt.broadcast %52 : tensor<64x1xf32> -> tensor<64x64xf32> 
    %54 = tt.load %35 : tensor<64x64x!tt.ptr<f32>> 
    %55 = tt.dot %45, %54, %53, inputPrecision = tf32 : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32> 
    %56 = arith.cmpf oeq, %41, %cst_3 : tensor<64xf32> 
    %57 = arith.select %56, %cst_2, %41 : tensor<64xi1>, tensor<64xf32> 
    %58 = arith.cmpf oeq, %57, %cst_2 : tensor<64xf32> 
    %59 = arith.select %58, %cst_1, %50 : tensor<64xi1>, tensor<64xf32> 
    %60 = tt.expand_dims %59 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> 
    %61 = tt.broadcast %60 : tensor<64x1xf32> -> tensor<64x64xf32> 
    %62 = arith.divf %55, %61 : tensor<64x64xf32> 
    %63 = tt.splat %arg19 : i32 -> tensor<64x1xi32> 
    %64 = arith.muli %21, %63 : tensor<64x1xi32> 
    %65 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>> 
    %66 = tt.addptr %65, %64 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32> 
    %67 = tt.splat %arg20 : i32 -> tensor<1x64xi32> 
    %68 = arith.muli %10, %67 : tensor<1x64xi32> 
    %69 = tt.broadcast %66 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>> 
    %70 = tt.broadcast %68 : tensor<1x64xi32> -> tensor<64x64xi32> 
    %71 = tt.addptr %69, %70 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32> 
    tt.store %71, %62 : tensor<64x64x!tt.ptr<f32>> 
    tt.return 
  } 
} 
