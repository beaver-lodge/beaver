defmodule Beaver.MLIR.CAPI.Handwritten do
  @moduledoc false

  # Arity is shared by each declaration group; the public and raw surfaces use
  # this same table, so their exports cannot drift apart.
  @functions for {arity, names} <- [
                   {4, ~w(
        apply_rewrite_pattern_set_with_module apply_rewrite_pattern_set_with_op type_converter_add_conversion_map
        type_converter_reply_callback type_converter_reply_types type_converter_reply_values optimization_remarks_attach
      )},
                   {2, ~w(
        callback_reply cuda_module_get_function cuda_memcpy_htod cuda_memcpy_dtoh operation_verify_async
        transform_state_payload_ops transform_state_payload_values transform_state_params
        context_attach_diagnostic_handler destroy_frozen_rewrite_pattern_set destroy_rewrite_pattern_set
        mlir_type_of_enif_obj read_opaque_ptr run_pm_on_op_async type_converter_convert_async
      )},
                   {3, ~w(
        callback_reply_code cuda_func_set_attribute module_create_parse_async compile_llvm_ir_to_ptx
        conversion_target_add_static jit_invoke_with_terms jit_invoke_with_terms_cpu_bound
        jit_invoke_with_terms_io_bound type_converter_reply_value value_replace_uses_with_if
      )},
                   {1, ~w(
        conditionally_speculatable_query_async memory_effects_query_async action_tracing_drain
        action_tracing_detach cuda_device_name cuda_device_compute_capability cuda_module_load cuda_module_unload
        cuda_mem_alloc cuda_mem_free triton_register_dialects external_interface_release
        translate_module_to_llvm_ir operation_inventory conversion_target_create conversion_target_destroy
        deallocate_opaque_ptr destroy_pm_async enif_signatures get_string_ref jit_register_enif
        memref_type_get_strides_and_offset registered_dialects registered_ops string_printer_flush
        to_string_AffineMap to_string_Attribute to_string_Diagnostic to_string_Identifier to_string_Location
        to_string_OpPassManager to_string_Operation to_string_OperationBytecode to_string_OperationGeneric
        to_string_OperationSpecialized to_string_StringRef to_string_Type to_string_Value type_converter_destroy
        unranked_memref_descriptor_deallocate_with_c unranked_memref_descriptor_deallocate_with_enif
        unranked_memref_descriptor_empty unranked_memref_descriptor_get_offset unranked_memref_descriptor_get_rank
        unranked_memref_descriptor_get_sizes unranked_memref_descriptor_get_strides
      )},
                   {5, ~w(
        action_tracing_attach create_mlir_rewrite_pattern
      )},
                   {0, ~w(
        cuda_available cuda_device_count cuda_synchronize triton_register_passes peak_rss process_cpu_time
        enif_functions get_null_ptr string_printer_callback type_converter_create
      )},
                   {9, ~w(
        cuda_launch_kernel
      )},
                   {7, ~w(
        infer_return_types infer_return_type_components
      )},
                   {8, ~w(
        compiler_kernel_load_and_populate apply_conversion_async
      )},
                   {6, ~w(
        create_mlir_pass
      )}
                 ],
                 name <- names,
                 do: {String.to_atom("beaver_raw_" <> name), arity}

  @doc false
  def functions, do: @functions
end
