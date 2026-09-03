module {
  tt.func public @gemma4_feature_width_layout_capability_probe(
    %query: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
    %prior_key: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
    %current_key: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
    %logits: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
    %START: i32,
    %TOKENS: i32
  ) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %c256 = arith.constant 256 : i32
    %c511 = arith.constant 511 : i32
    %c512 = arith.constant 512 : i32
    %c2048 = arith.constant 2048 : i32
    %c262656 = arith.constant 262656 : i32
    %negative = arith.constant dense<-3.400000e+38> : tensor<16x64xf32>
    %query_zeros = arith.constant dense<0.000000e+00> : tensor<16x256xbf16>
    %key_zeros = arith.constant dense<0.000000e+00> : tensor<256x64xbf16>
    %program = tt.get_program_id x : i32
    %query_block = arith.divsi %program, %c8 : i32
    %query_head = arith.remsi %program, %c8 : i32
    %kv_head = arith.divsi %query_head, %c4 : i32
    %query_start = arith.muli %query_block, %c16 : i32
    %query_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %dim_range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %key_range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %query_start_vector = tt.splat %query_start : i32 -> tensor<16xi32>
    %query_rows = arith.addi %query_start_vector, %query_range : tensor<16xi32>
    %tokens_vector = tt.splat %TOKENS : i32 -> tensor<16xi32>
    %valid_queries = arith.cmpi slt, %query_rows, %tokens_vector : tensor<16xi32>
    %start_vector = tt.splat %START : i32 -> tensor<16xi32>
    %query_positions = arith.addi %start_vector, %query_rows : tensor<16xi32>
    %window_offset = tt.splat %c511 : i32 -> tensor<16xi32>
    %query_first_candidate = arith.subi %query_positions, %window_offset : tensor<16xi32>
    %zero_vector = tt.splat %c0 : i32 -> tensor<16xi32>
    %query_first = arith.maxsi %query_first_candidate, %zero_vector : tensor<16xi32>

    %query_row_column = tt.expand_dims %query_rows {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %query_stride = tt.splat %c2048 : i32 -> tensor<16x1xi32>
    %query_row_offsets = arith.muli %query_row_column, %query_stride : tensor<16x1xi32>
    %query_head_scalar = arith.muli %query_head, %c256 : i32
    %query_head_column = tt.splat %query_head_scalar : i32 -> tensor<16x1xi32>
    %query_head_offsets = arith.addi %query_row_offsets, %query_head_column : tensor<16x1xi32>
    %query_rows_matrix = tt.broadcast %query_head_offsets : tensor<16x1xi32> -> tensor<16x256xi32>
    %dim_row = tt.expand_dims %dim_range {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %dims_matrix = tt.broadcast %dim_row : tensor<1x256xi32> -> tensor<16x256xi32>
    %query_offsets = arith.addi %query_rows_matrix, %dims_matrix : tensor<16x256xi32>
    %query_mask_column = tt.expand_dims %valid_queries {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %query_mask = tt.broadcast %query_mask_column : tensor<16x1xi1> -> tensor<16x256xi1>
    %query_base = tt.splat %query : !tt.ptr<bf16> -> tensor<16x256x!tt.ptr<bf16>>
    %query_ptrs = tt.addptr %query_base, %query_offsets : tensor<16x256x!tt.ptr<bf16>>, tensor<16x256xi32>
    %queries = tt.load %query_ptrs, %query_mask, %query_zeros : tensor<16x256x!tt.ptr<bf16>>

    %first_position = arith.addi %START, %query_start : i32
    %candidate_first = arith.subi %first_position, %c511 : i32
    %first_before_zero = arith.cmpi slt, %candidate_first, %c0 : i32
    %first = arith.select %first_before_zero, %c0, %candidate_first : i32
    %candidate_limit0 = arith.addi %query_start, %c16 : i32
    %past_tokens = arith.cmpi sgt, %candidate_limit0, %TOKENS : i32
    %candidate_limit = arith.select %past_tokens, %TOKENS, %candidate_limit0 : i32
    %limit = arith.addi %START, %candidate_limit : i32
    %kv_head_scalar = arith.muli %kv_head, %c256 : i32
    %scratch_head = arith.muli %query_head, %c262656 : i32

    scf.for %key_start = %first to %limit step %c64 : i32 {
      %key_start_vector = tt.splat %key_start : i32 -> tensor<64xi32>
      %key_tokens = arith.addi %key_start_vector, %key_range : tensor<64xi32>
      %limit_vector = tt.splat %limit : i32 -> tensor<64xi32>
      %within_limit = arith.cmpi slt, %key_tokens, %limit_vector : tensor<64xi32>
      %start_kv_vector = tt.splat %START : i32 -> tensor<64xi32>
      %is_current = arith.cmpi sge, %key_tokens, %start_kv_vector : tensor<64xi32>
      %is_prior = arith.cmpi slt, %key_tokens, %start_kv_vector : tensor<64xi32>
      %prior_valid = arith.andi %within_limit, %is_prior : tensor<64xi1>
      %current_valid = arith.andi %within_limit, %is_current : tensor<64xi1>
      %ring_modulus = tt.splat %c512 : i32 -> tensor<64xi32>
      %ring_rows = arith.remsi %key_tokens, %ring_modulus : tensor<64xi32>
      %current_rows = arith.subi %key_tokens, %start_kv_vector : tensor<64xi32>

      %query_positions_column = tt.expand_dims %query_positions {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %query_positions_matrix = tt.broadcast %query_positions_column : tensor<16x1xi32> -> tensor<16x64xi32>
      %key_tokens_row = tt.expand_dims %key_tokens {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %key_tokens_matrix = tt.broadcast %key_tokens_row : tensor<1x64xi32> -> tensor<16x64xi32>
      %causal = arith.cmpi sle, %key_tokens_matrix, %query_positions_matrix : tensor<16x64xi32>
      %query_first_column = tt.expand_dims %query_first {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %query_first_matrix = tt.broadcast %query_first_column : tensor<16x1xi32> -> tensor<16x64xi32>
      %within_window = arith.cmpi sge, %key_tokens_matrix, %query_first_matrix : tensor<16x64xi32>
      %key_valid_row = tt.expand_dims %within_limit {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
      %key_valid_matrix = tt.broadcast %key_valid_row : tensor<1x64xi1> -> tensor<16x64xi1>
      %query_valid_matrix = tt.broadcast %query_mask_column : tensor<16x1xi1> -> tensor<16x64xi1>
      %valid0 = arith.andi %causal, %within_window : tensor<16x64xi1>
      %valid1 = arith.andi %valid0, %key_valid_matrix : tensor<16x64xi1>
      %valid = arith.andi %valid1, %query_valid_matrix : tensor<16x64xi1>

      %ring_row = tt.expand_dims %ring_rows {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %ring_stride = tt.splat %c512 : i32 -> tensor<1x64xi32>
      %prior_row_offsets = arith.muli %ring_row, %ring_stride : tensor<1x64xi32>
      %current_row = tt.expand_dims %current_rows {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %current_row_offsets = arith.muli %current_row, %ring_stride : tensor<1x64xi32>
      %kv_head_row = tt.splat %kv_head_scalar : i32 -> tensor<1x64xi32>
      %prior_token_offsets = arith.addi %prior_row_offsets, %kv_head_row : tensor<1x64xi32>
      %current_token_offsets = arith.addi %current_row_offsets, %kv_head_row : tensor<1x64xi32>
      %prior_columns = tt.broadcast %prior_token_offsets : tensor<1x64xi32> -> tensor<256x64xi32>
      %current_columns = tt.broadcast %current_token_offsets : tensor<1x64xi32> -> tensor<256x64xi32>
      %dim_column = tt.expand_dims %dim_range {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
      %key_dims = tt.broadcast %dim_column : tensor<256x1xi32> -> tensor<256x64xi32>
      %prior_offsets = arith.addi %prior_columns, %key_dims : tensor<256x64xi32>
      %current_offsets = arith.addi %current_columns, %key_dims : tensor<256x64xi32>
      %prior_mask_row = tt.expand_dims %prior_valid {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
      %prior_mask = tt.broadcast %prior_mask_row : tensor<1x64xi1> -> tensor<256x64xi1>
      %current_mask_row = tt.expand_dims %current_valid {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
      %current_mask = tt.broadcast %current_mask_row : tensor<1x64xi1> -> tensor<256x64xi1>
      %prior_base = tt.splat %prior_key : !tt.ptr<bf16> -> tensor<256x64x!tt.ptr<bf16>>
      %prior_ptrs = tt.addptr %prior_base, %prior_offsets : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
      %prior_keys = tt.load %prior_ptrs, %prior_mask, %key_zeros : tensor<256x64x!tt.ptr<bf16>>
      %current_base = tt.splat %current_key : !tt.ptr<bf16> -> tensor<256x64x!tt.ptr<bf16>>
      %current_ptrs = tt.addptr %current_base, %current_offsets : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
      %current_keys = tt.load %current_ptrs, %current_mask, %key_zeros : tensor<256x64x!tt.ptr<bf16>>
      %keys = arith.addf %prior_keys, %current_keys : tensor<256x64xbf16>
      %dot0 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
      %raw_scores = tt.dot %queries, %keys, %dot0, inputPrecision = tf32 : tensor<16x256xbf16> * tensor<256x64xbf16> -> tensor<16x64xf32>
      %scores = arith.select %valid, %raw_scores, %negative : tensor<16x64xi1>, tensor<16x64xf32>
      %scores_bf16 = arith.truncf %scores : tensor<16x64xf32> to tensor<16x64xbf16>

      %row_stride = tt.splat %c512 : i32 -> tensor<16x1xi32>
      %scratch_rows = arith.muli %query_row_column, %row_stride : tensor<16x1xi32>
      %scratch_head_matrix = tt.splat %scratch_head : i32 -> tensor<16x1xi32>
      %scratch_row_base = arith.addi %scratch_rows, %scratch_head_matrix : tensor<16x1xi32>
      %scratch_row_matrix = tt.broadcast %scratch_row_base : tensor<16x1xi32> -> tensor<16x64xi32>
      %local_columns = arith.subi %key_tokens_matrix, %query_first_matrix : tensor<16x64xi32>
      %scratch_offsets = arith.addi %scratch_row_matrix, %local_columns : tensor<16x64xi32>
      %logits_base = tt.splat %logits : !tt.ptr<bf16> -> tensor<16x64x!tt.ptr<bf16>>
      %logits_ptrs = tt.addptr %logits_base, %scratch_offsets : tensor<16x64x!tt.ptr<bf16>>, tensor<16x64xi32>
      tt.store %logits_ptrs, %scores_bf16, %valid : tensor<16x64x!tt.ptr<bf16>>
    }
    tt.return
  }
}
