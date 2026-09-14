"""The C ABI, as ctypes sees it.

Generated from ``src/csrc/include/bartorch.h`` by ``scripts/gen_abi.py``.
Do not edit: run the generator instead.  ``tests/test_abi.py`` fails when
this file is not what the header produces.
"""

from __future__ import annotations

import ctypes

#: Dimensions BART carries for every array.
DIMS = 16

#: BART's debug levels, by the header's own names.
LOG_LEVELS = {
    "error": 0,
    "warn": 1,
    "info": 2,
    "debug1": 3,
    "debug2": 4,
    "debug3": 5,
    "debug4": 6,
    "trace": 7,
}

ALLOC_FN = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_long),
)
FREE_FN = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_void_p,
)
LOG_FN = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
)
APPLY_FN = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
)
GENERIC_APPLY_FN = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
)
PAIR_APPLY_FN = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
)
RELEASE_FN = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
)

#: Every entry point the header exports.
SYMBOLS = (
    "bartorch_bart_version",
    "bartorch_build_info",
    "bartorch_set_allocator",
    "bartorch_set_log_handler",
    "bartorch_set_debug_level",
    "bartorch_get_debug_level",
    "bartorch_set_num_threads",
    "bartorch_register",
    "bartorch_exists",
    "bartorch_lookup",
    "bartorch_unlink",
    "bartorch_unlink_all",
    "bartorch_command",
    "bartorch_backend_set",
    "bartorch_backend_count",
    "bartorch_backend_name",
    "bartorch_backend_has_fallback",
    "bartorch_nufft_set_stream_psf",
    "bartorch_nufft_stream_psf",
    "bartorch_nufft_set_compress_psf",
    "bartorch_nufft_compress_psf",
    "bartorch_nufft_set_overlap_psf",
    "bartorch_nufft_overlap_psf",
    "bartorch_nufft_set_contraction_kernel",
    "bartorch_nufft_set_release_transforms",
    "bartorch_nufft_release_transforms",
    "bartorch_nufft_set_fft_callbacks",
    "bartorch_nufft_fft_callbacks",
    "bartorch_nufft_set_paired",
    "bartorch_nufft_paired",
    "bartorch_nufft_paired_built",
    "bartorch_nufft_set_bf16",
    "bartorch_nufft_bf16",
    "bartorch_host_alloc",
    "bartorch_host_free",
    "bartorch_cuda_stage_open",
    "bartorch_cuda_stage_close",
    "bartorch_cuda_stage_copy",
    "bartorch_cuda_stage_wait",
    "bartorch_cuda_stage_release",
    "bartorch_cuda_copy_pageable",
    "bartorch_host_prefault_begin",
    "bartorch_host_prefault_end",
    "bartorch_cuda_host_register",
    "bartorch_cuda_host_unregister",
    "bartorch_sense_set_coil_batch",
    "bartorch_sense_coil_batch",
    "bartorch_sense_set_fold_maps",
    "bartorch_sense_fold_maps",
    "bartorch_sense_counter",
    "bartorch_sense_reset_counters",
    "bartorch_fft_set",
    "bartorch_fft_usable",
    "bartorch_fft_counter",
    "bartorch_fft_reset_counters",
    "bartorch_linop_callback",
    "bartorch_linop_fft",
    "bartorch_linop_cdiag",
    "bartorch_linop_fmac",
    "bartorch_linop_sampling",
    "bartorch_linop_nufft",
    "bartorch_linop_sense",
    "bartorch_linop_cartesian",
    "bartorch_grid_fused",
    "bartorch_linop_coils",
    "bartorch_linop_with_normal",
    "bartorch_linop_chain",
    "bartorch_linop_plus",
    "bartorch_linop_adjoint_op",
    "bartorch_linop_stack_cod",
    "bartorch_linop_stack",
    "bartorch_linop_normal_op",
    "bartorch_linop_scale",
    "bartorch_linop_zconj",
    "bartorch_linop_identity",
    "bartorch_linop_null",
    "bartorch_linop_maxeigen",
    "bartorch_linop_zreal",
    "bartorch_linop_rdiag",
    "bartorch_linop_matrix",
    "bartorch_linop_conv",
    "bartorch_linop_grad",
    "bartorch_linop_sum",
    "bartorch_linop_scaled_sum",
    "bartorch_linop_avg",
    "bartorch_linop_repmat",
    "bartorch_linop_flip",
    "bartorch_linop_hankel",
    "bartorch_linop_reshape",
    "bartorch_linop_resize",
    "bartorch_linop_extract",
    "bartorch_linop_transpose",
    "bartorch_linop_permute",
    "bartorch_linop_shift",
    "bartorch_linop_padding",
    "bartorch_linop_has_pseudo_inv",
    "bartorch_linop_pseudo_inv",
    "bartorch_linop_domain",
    "bartorch_linop_codomain",
    "bartorch_linop_forward",
    "bartorch_linop_adjoint",
    "bartorch_linop_normal",
    "bartorch_linop_free",
    "bartorch_maxeigen",
    "bartorch_solve",
    "bartorch_solve_error",
    "bartorch_scaling_norm",
    "bartorch_prox_create",
    "bartorch_prox_domain",
    "bartorch_prox_apply",
    "bartorch_prox_transform_apply",
    "bartorch_prox_transform",
    "bartorch_prox_transform_is_identity",
    "bartorch_prox_rewind",
    "bartorch_prox_free",
    "bartorch_nlop_callback",
    "bartorch_nlop_callback_generic",
    "bartorch_nlop_from_linop",
    "bartorch_nlop_chain",
    "bartorch_nlop_domain",
    "bartorch_nlop_codomain",
    "bartorch_nlop_apply",
    "bartorch_nlop_derivative",
    "bartorch_nlop_adjoint",
    "bartorch_nlop_inputs",
    "bartorch_nlop_outputs",
    "bartorch_nlop_input_domain",
    "bartorch_nlop_output_codomain",
    "bartorch_nlop_apply_generic",
    "bartorch_nlop_derivative_linop",
    "bartorch_nlop_reshape_in",
    "bartorch_nlop_reshape_out",
    "bartorch_nlop_chain2",
    "bartorch_nlop_combine",
    "bartorch_nlop_link",
    "bartorch_nlop_dup",
    "bartorch_nlop_stack_inputs",
    "bartorch_nlop_stack_outputs",
    "bartorch_nlop_permute",
    "bartorch_nlop_del_out",
    "bartorch_nlop_flatten",
    "bartorch_nlop_tenmul",
    "bartorch_nlop_zdiv",
    "bartorch_nlop_zaxpbz",
    "bartorch_nlop_zexp",
    "bartorch_nlop_zlog",
    "bartorch_nlop_zinv",
    "bartorch_nlop_zsqrt",
    "bartorch_nlop_zspow",
    "bartorch_nlop_zsadd",
    "bartorch_nlop_zabs",
    "bartorch_nlop_smo_abs",
    "bartorch_nlop_zrss",
    "bartorch_nlop_zss",
    "bartorch_nlop_const",
    "bartorch_nlop_set_input_const",
    "bartorch_nlop_free",
    "bartorch_noir_create",
    "bartorch_noir_model",
    "bartorch_noir_coils",
    "bartorch_noir_image",
    "bartorch_noir_data",
    "bartorch_noir_transform",
    "bartorch_noir_dims",
    "bartorch_noir_free",
    "bartorch_noir_net_create",
    "bartorch_noir_net_step",
    "bartorch_noir_net_iterations",
    "bartorch_noir_net_adjoint",
    "bartorch_noir_net_decompose",
    "bartorch_noir_net_split",
    "bartorch_noir_net_join",
    "bartorch_noir_net_free",
    "bartorch_irgnm",
    "bartorch_irgnm2",
    "bartorch_cuda_built",
    "bartorch_cuda_device_count",
    "bartorch_cuda_enable",
    "bartorch_cuda_device",
    "bartorch_cuda_set_streams",
    "bartorch_cuda_get_streams",
    "bartorch_cuda_use_memcache",
    "bartorch_cuda_memcache_clear_all",
    "bartorch_cuda_wait_for_stream",
    "bartorch_cuda_signal_stream",
    "bartorch_cuda_free_memory",
    "bartorch_finufft_set",
    "bartorch_finufft_layout",
    "bartorch_finufft_set_tolerance",
    "bartorch_finufft_tolerance",
    "bartorch_finufft_set_upsampling",
    "bartorch_finufft_upsampling",
    "bartorch_finufft_set_threads",
    "bartorch_finufft_threads",
    "bartorch_finufft_use_in_tools",
    "bartorch_finufft_usable_on",
    "bartorch_finufft_usable",
    "bartorch_finufft_live_plans",
    "bartorch_last_error",
    "bartorch_clear_error",
    "bartorch_nufft_decline_reason",
    "bartorch_nufft_decline_text",
    "bartorch_nufft_allow_fallback",
    "bartorch_nufft_fallback_allowed",
    "bartorch_nufft_counter",
    "bartorch_nufft_reset_counters",
    "bartorch_toeplitz_counter",
    "bartorch_toeplitz_reset_counters",
    "bartorch_on_device",
)


def bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Give every entry point its signature, and return the library."""
    lib.bartorch_bart_version.restype = ctypes.c_char_p
    lib.bartorch_bart_version.argtypes = []
    lib.bartorch_build_info.restype = ctypes.c_char_p
    lib.bartorch_build_info.argtypes = []
    lib.bartorch_set_allocator.restype = None
    lib.bartorch_set_allocator.argtypes = [ALLOC_FN, FREE_FN, ctypes.c_void_p]
    lib.bartorch_set_log_handler.restype = None
    lib.bartorch_set_log_handler.argtypes = [LOG_FN, ctypes.c_void_p]
    lib.bartorch_set_debug_level.restype = None
    lib.bartorch_set_debug_level.argtypes = [ctypes.c_int]
    lib.bartorch_get_debug_level.restype = ctypes.c_int
    lib.bartorch_get_debug_level.argtypes = []
    lib.bartorch_set_num_threads.restype = None
    lib.bartorch_set_num_threads.argtypes = [ctypes.c_int]
    lib.bartorch_register.restype = ctypes.c_int
    lib.bartorch_register.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
    ]
    lib.bartorch_exists.restype = ctypes.c_int
    lib.bartorch_exists.argtypes = [ctypes.c_char_p]
    lib.bartorch_lookup.restype = ctypes.c_int
    lib.bartorch_lookup.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.bartorch_unlink.restype = ctypes.c_int
    lib.bartorch_unlink.argtypes = [ctypes.c_char_p]
    lib.bartorch_unlink_all.restype = ctypes.c_int
    lib.bartorch_unlink_all.argtypes = []
    lib.bartorch_command.restype = ctypes.c_int
    lib.bartorch_command.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.bartorch_backend_set.restype = ctypes.c_int
    lib.bartorch_backend_set.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    lib.bartorch_backend_count.restype = ctypes.c_int
    lib.bartorch_backend_count.argtypes = []
    lib.bartorch_backend_name.restype = ctypes.c_char_p
    lib.bartorch_backend_name.argtypes = [ctypes.c_int]
    lib.bartorch_backend_has_fallback.restype = ctypes.c_int
    lib.bartorch_backend_has_fallback.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_set_stream_psf.restype = None
    lib.bartorch_nufft_set_stream_psf.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_stream_psf.restype = ctypes.c_int
    lib.bartorch_nufft_stream_psf.argtypes = []
    lib.bartorch_nufft_set_compress_psf.restype = None
    lib.bartorch_nufft_set_compress_psf.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_compress_psf.restype = ctypes.c_int
    lib.bartorch_nufft_compress_psf.argtypes = []
    lib.bartorch_nufft_set_overlap_psf.restype = None
    lib.bartorch_nufft_set_overlap_psf.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_overlap_psf.restype = ctypes.c_int
    lib.bartorch_nufft_overlap_psf.argtypes = []
    lib.bartorch_nufft_set_contraction_kernel.restype = None
    lib.bartorch_nufft_set_contraction_kernel.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_set_release_transforms.restype = None
    lib.bartorch_nufft_set_release_transforms.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_release_transforms.restype = ctypes.c_int
    lib.bartorch_nufft_release_transforms.argtypes = []
    lib.bartorch_nufft_set_fft_callbacks.restype = None
    lib.bartorch_nufft_set_fft_callbacks.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_fft_callbacks.restype = ctypes.c_int
    lib.bartorch_nufft_fft_callbacks.argtypes = []
    lib.bartorch_nufft_set_paired.restype = None
    lib.bartorch_nufft_set_paired.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_paired.restype = ctypes.c_int
    lib.bartorch_nufft_paired.argtypes = []
    lib.bartorch_nufft_paired_built.restype = ctypes.c_int
    lib.bartorch_nufft_paired_built.argtypes = []
    lib.bartorch_nufft_set_bf16.restype = None
    lib.bartorch_nufft_set_bf16.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_bf16.restype = ctypes.c_int
    lib.bartorch_nufft_bf16.argtypes = []
    lib.bartorch_host_alloc.restype = ctypes.c_void_p
    lib.bartorch_host_alloc.argtypes = [ctypes.c_long, ctypes.c_int]
    lib.bartorch_host_free.restype = None
    lib.bartorch_host_free.argtypes = [ctypes.c_void_p]
    lib.bartorch_cuda_stage_open.restype = ctypes.c_int
    lib.bartorch_cuda_stage_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.bartorch_cuda_stage_close.restype = None
    lib.bartorch_cuda_stage_close.argtypes = [ctypes.c_void_p]
    lib.bartorch_cuda_stage_copy.restype = ctypes.c_int
    lib.bartorch_cuda_stage_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_long,
    ]
    lib.bartorch_cuda_stage_wait.restype = ctypes.c_int
    lib.bartorch_cuda_stage_wait.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.bartorch_cuda_stage_release.restype = ctypes.c_int
    lib.bartorch_cuda_stage_release.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.bartorch_cuda_copy_pageable.restype = ctypes.c_int
    lib.bartorch_cuda_copy_pageable.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
    lib.bartorch_host_prefault_begin.restype = ctypes.c_void_p
    lib.bartorch_host_prefault_begin.argtypes = [ctypes.c_void_p, ctypes.c_long]
    lib.bartorch_host_prefault_end.restype = None
    lib.bartorch_host_prefault_end.argtypes = [ctypes.c_void_p]
    lib.bartorch_cuda_host_register.restype = ctypes.c_int
    lib.bartorch_cuda_host_register.argtypes = [ctypes.c_void_p, ctypes.c_long]
    lib.bartorch_cuda_host_unregister.restype = None
    lib.bartorch_cuda_host_unregister.argtypes = [ctypes.c_void_p]
    lib.bartorch_sense_set_coil_batch.restype = None
    lib.bartorch_sense_set_coil_batch.argtypes = [ctypes.c_int]
    lib.bartorch_sense_coil_batch.restype = ctypes.c_int
    lib.bartorch_sense_coil_batch.argtypes = []
    lib.bartorch_sense_set_fold_maps.restype = None
    lib.bartorch_sense_set_fold_maps.argtypes = [ctypes.c_int]
    lib.bartorch_sense_fold_maps.restype = ctypes.c_int
    lib.bartorch_sense_fold_maps.argtypes = []
    lib.bartorch_sense_counter.restype = ctypes.c_long
    lib.bartorch_sense_counter.argtypes = [ctypes.c_int]
    lib.bartorch_sense_reset_counters.restype = None
    lib.bartorch_sense_reset_counters.argtypes = []
    lib.bartorch_fft_set.restype = ctypes.c_int
    lib.bartorch_fft_set.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    lib.bartorch_fft_usable.restype = ctypes.c_int
    lib.bartorch_fft_usable.argtypes = []
    lib.bartorch_fft_counter.restype = ctypes.c_long
    lib.bartorch_fft_counter.argtypes = [ctypes.c_int]
    lib.bartorch_fft_reset_counters.restype = None
    lib.bartorch_fft_reset_counters.argtypes = []
    lib.bartorch_linop_callback.restype = ctypes.c_void_p
    lib.bartorch_linop_callback.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        APPLY_FN,
        APPLY_FN,
        APPLY_FN,
        ctypes.c_void_p,
        RELEASE_FN,
    ]
    lib.bartorch_linop_fft.restype = ctypes.c_void_p
    lib.bartorch_linop_fft.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.bartorch_linop_cdiag.restype = ctypes.c_void_p
    lib.bartorch_linop_cdiag.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    lib.bartorch_linop_fmac.restype = ctypes.c_void_p
    lib.bartorch_linop_fmac.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
    ]
    lib.bartorch_linop_sampling.restype = ctypes.c_void_p
    lib.bartorch_linop_sampling.argtypes = [
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
    ]
    lib.bartorch_linop_nufft.restype = ctypes.c_void_p
    lib.bartorch_linop_nufft.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.bartorch_linop_sense.restype = ctypes.c_void_p
    lib.bartorch_linop_sense.argtypes = [
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.bartorch_linop_cartesian.restype = ctypes.c_void_p
    lib.bartorch_linop_cartesian.argtypes = [
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
    ]
    lib.bartorch_grid_fused.restype = ctypes.c_long
    lib.bartorch_grid_fused.argtypes = []
    lib.bartorch_linop_coils.restype = ctypes.c_void_p
    lib.bartorch_linop_coils.argtypes = [
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.bartorch_linop_with_normal.restype = ctypes.c_void_p
    lib.bartorch_linop_with_normal.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_linop_chain.restype = ctypes.c_void_p
    lib.bartorch_linop_chain.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_linop_plus.restype = ctypes.c_void_p
    lib.bartorch_linop_plus.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_linop_adjoint_op.restype = ctypes.c_void_p
    lib.bartorch_linop_adjoint_op.argtypes = [ctypes.c_void_p]
    lib.bartorch_linop_stack_cod.restype = ctypes.c_void_p
    lib.bartorch_linop_stack_cod.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    lib.bartorch_linop_stack.restype = ctypes.c_void_p
    lib.bartorch_linop_stack.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.bartorch_linop_normal_op.restype = ctypes.c_void_p
    lib.bartorch_linop_normal_op.argtypes = [ctypes.c_void_p]
    lib.bartorch_linop_scale.restype = ctypes.c_void_p
    lib.bartorch_linop_scale.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.bartorch_linop_zconj.restype = ctypes.c_void_p
    lib.bartorch_linop_zconj.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
    lib.bartorch_linop_identity.restype = ctypes.c_void_p
    lib.bartorch_linop_identity.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
    lib.bartorch_linop_null.restype = ctypes.c_void_p
    lib.bartorch_linop_null.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_linop_maxeigen.restype = ctypes.c_double
    lib.bartorch_linop_maxeigen.argtypes = [ctypes.c_void_p]
    lib.bartorch_linop_zreal.restype = ctypes.c_void_p
    lib.bartorch_linop_zreal.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
    lib.bartorch_linop_rdiag.restype = ctypes.c_void_p
    lib.bartorch_linop_rdiag.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    lib.bartorch_linop_matrix.restype = ctypes.c_void_p
    lib.bartorch_linop_matrix.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
    ]
    lib.bartorch_linop_conv.restype = ctypes.c_void_p
    lib.bartorch_linop_conv.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
    ]
    lib.bartorch_linop_grad.restype = ctypes.c_void_p
    lib.bartorch_linop_grad.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
        ctypes.c_ulong,
    ]
    lib.bartorch_linop_sum.restype = ctypes.c_void_p
    lib.bartorch_linop_sum.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_ulong]
    lib.bartorch_linop_scaled_sum.restype = ctypes.c_void_p
    lib.bartorch_linop_scaled_sum.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_ulong,
    ]
    lib.bartorch_linop_avg.restype = ctypes.c_void_p
    lib.bartorch_linop_avg.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_ulong]
    lib.bartorch_linop_repmat.restype = ctypes.c_void_p
    lib.bartorch_linop_repmat.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_ulong,
    ]
    lib.bartorch_linop_flip.restype = ctypes.c_void_p
    lib.bartorch_linop_flip.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_ulong]
    lib.bartorch_linop_hankel.restype = ctypes.c_void_p
    lib.bartorch_linop_hankel.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.bartorch_linop_reshape.restype = ctypes.c_void_p
    lib.bartorch_linop_reshape.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_linop_resize.restype = ctypes.c_void_p
    lib.bartorch_linop_resize.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_linop_extract.restype = ctypes.c_void_p
    lib.bartorch_linop_extract.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_linop_transpose.restype = ctypes.c_void_p
    lib.bartorch_linop_transpose.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_linop_permute.restype = ctypes.c_void_p
    lib.bartorch_linop_permute.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_linop_shift.restype = ctypes.c_void_p
    lib.bartorch_linop_shift.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
        ctypes.c_long,
        ctypes.c_int,
    ]
    lib.bartorch_linop_padding.restype = ctypes.c_void_p
    lib.bartorch_linop_padding.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_linop_has_pseudo_inv.restype = ctypes.c_int
    lib.bartorch_linop_has_pseudo_inv.argtypes = [ctypes.c_void_p]
    lib.bartorch_linop_pseudo_inv.restype = ctypes.c_int
    lib.bartorch_linop_pseudo_inv.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.bartorch_linop_domain.restype = ctypes.c_int
    lib.bartorch_linop_domain.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_linop_codomain.restype = ctypes.c_int
    lib.bartorch_linop_codomain.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_linop_forward.restype = ctypes.c_int
    lib.bartorch_linop_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_linop_adjoint.restype = ctypes.c_int
    lib.bartorch_linop_adjoint.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_linop_normal.restype = ctypes.c_int
    lib.bartorch_linop_normal.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_linop_free.restype = None
    lib.bartorch_linop_free.argtypes = [ctypes.c_void_p]
    lib.bartorch_maxeigen.restype = ctypes.c_int
    lib.bartorch_maxeigen.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.bartorch_solve.restype = ctypes.c_int
    lib.bartorch_solve.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_solve_error.restype = ctypes.c_char_p
    lib.bartorch_solve_error.argtypes = [ctypes.c_int]
    lib.bartorch_scaling_norm.restype = ctypes.c_float
    lib.bartorch_scaling_norm.argtypes = [
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.bartorch_prox_create.restype = ctypes.c_int
    lib.bartorch_prox_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.bartorch_prox_domain.restype = ctypes.c_int
    lib.bartorch_prox_domain.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_prox_apply.restype = ctypes.c_int
    lib.bartorch_prox_apply.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.bartorch_prox_transform_apply.restype = ctypes.c_int
    lib.bartorch_prox_transform_apply.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.bartorch_prox_transform.restype = ctypes.c_void_p
    lib.bartorch_prox_transform.argtypes = [ctypes.c_void_p]
    lib.bartorch_prox_transform_is_identity.restype = ctypes.c_int
    lib.bartorch_prox_transform_is_identity.argtypes = [ctypes.c_void_p]
    lib.bartorch_prox_rewind.restype = ctypes.c_int
    lib.bartorch_prox_rewind.argtypes = [ctypes.c_void_p]
    lib.bartorch_prox_free.restype = None
    lib.bartorch_prox_free.argtypes = [ctypes.c_void_p]
    lib.bartorch_nlop_callback.restype = ctypes.c_void_p
    lib.bartorch_nlop_callback.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        APPLY_FN,
        APPLY_FN,
        APPLY_FN,
        ctypes.c_void_p,
        RELEASE_FN,
    ]
    lib.bartorch_nlop_callback_generic.restype = ctypes.c_void_p
    lib.bartorch_nlop_callback_generic.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        GENERIC_APPLY_FN,
        PAIR_APPLY_FN,
        PAIR_APPLY_FN,
        ctypes.c_void_p,
        RELEASE_FN,
    ]
    lib.bartorch_nlop_from_linop.restype = ctypes.c_void_p
    lib.bartorch_nlop_from_linop.argtypes = [ctypes.c_void_p]
    lib.bartorch_nlop_chain.restype = ctypes.c_void_p
    lib.bartorch_nlop_chain.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_nlop_domain.restype = ctypes.c_int
    lib.bartorch_nlop_domain.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_nlop_codomain.restype = ctypes.c_int
    lib.bartorch_nlop_codomain.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_nlop_apply.restype = ctypes.c_int
    lib.bartorch_nlop_apply.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_nlop_derivative.restype = ctypes.c_int
    lib.bartorch_nlop_derivative.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_nlop_adjoint.restype = ctypes.c_int
    lib.bartorch_nlop_adjoint.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_nlop_inputs.restype = ctypes.c_int
    lib.bartorch_nlop_inputs.argtypes = [ctypes.c_void_p]
    lib.bartorch_nlop_outputs.restype = ctypes.c_int
    lib.bartorch_nlop_outputs.argtypes = [ctypes.c_void_p]
    lib.bartorch_nlop_input_domain.restype = ctypes.c_int
    lib.bartorch_nlop_input_domain.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_nlop_output_codomain.restype = ctypes.c_int
    lib.bartorch_nlop_output_codomain.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_nlop_apply_generic.restype = ctypes.c_int
    lib.bartorch_nlop_apply_generic.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.bartorch_nlop_derivative_linop.restype = ctypes.c_void_p
    lib.bartorch_nlop_derivative_linop.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.bartorch_nlop_reshape_in.restype = ctypes.c_void_p
    lib.bartorch_nlop_reshape_in.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_nlop_reshape_out.restype = ctypes.c_void_p
    lib.bartorch_nlop_reshape_out.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_nlop_chain2.restype = ctypes.c_void_p
    lib.bartorch_nlop_chain2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.bartorch_nlop_combine.restype = ctypes.c_void_p
    lib.bartorch_nlop_combine.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.bartorch_nlop_link.restype = ctypes.c_void_p
    lib.bartorch_nlop_link.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.bartorch_nlop_dup.restype = ctypes.c_void_p
    lib.bartorch_nlop_dup.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.bartorch_nlop_stack_inputs.restype = ctypes.c_void_p
    lib.bartorch_nlop_stack_inputs.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.bartorch_nlop_stack_outputs.restype = ctypes.c_void_p
    lib.bartorch_nlop_stack_outputs.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.bartorch_nlop_permute.restype = ctypes.c_void_p
    lib.bartorch_nlop_permute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.bartorch_nlop_del_out.restype = ctypes.c_void_p
    lib.bartorch_nlop_del_out.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.bartorch_nlop_flatten.restype = ctypes.c_void_p
    lib.bartorch_nlop_flatten.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.bartorch_nlop_tenmul.restype = ctypes.c_void_p
    lib.bartorch_nlop_tenmul.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_nlop_zdiv.restype = ctypes.c_void_p
    lib.bartorch_nlop_zdiv.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_float]
    lib.bartorch_nlop_zaxpbz.restype = ctypes.c_void_p
    lib.bartorch_nlop_zaxpbz.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.bartorch_nlop_zexp.restype = ctypes.c_void_p
    lib.bartorch_nlop_zexp.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
    lib.bartorch_nlop_zlog.restype = ctypes.c_void_p
    lib.bartorch_nlop_zlog.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
    lib.bartorch_nlop_zinv.restype = ctypes.c_void_p
    lib.bartorch_nlop_zinv.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_float]
    lib.bartorch_nlop_zsqrt.restype = ctypes.c_void_p
    lib.bartorch_nlop_zsqrt.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
    lib.bartorch_nlop_zspow.restype = ctypes.c_void_p
    lib.bartorch_nlop_zspow.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.bartorch_nlop_zsadd.restype = ctypes.c_void_p
    lib.bartorch_nlop_zsadd.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.bartorch_nlop_zabs.restype = ctypes.c_void_p
    lib.bartorch_nlop_zabs.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
    lib.bartorch_nlop_smo_abs.restype = ctypes.c_void_p
    lib.bartorch_nlop_smo_abs.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_float,
    ]
    lib.bartorch_nlop_zrss.restype = ctypes.c_void_p
    lib.bartorch_nlop_zrss.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_ulong,
        ctypes.c_float,
    ]
    lib.bartorch_nlop_zss.restype = ctypes.c_void_p
    lib.bartorch_nlop_zss.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_ulong]
    lib.bartorch_nlop_const.restype = ctypes.c_void_p
    lib.bartorch_nlop_const.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
    ]
    lib.bartorch_nlop_set_input_const.restype = ctypes.c_void_p
    lib.bartorch_nlop_set_input_const.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
    ]
    lib.bartorch_nlop_free.restype = None
    lib.bartorch_nlop_free.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_create.restype = ctypes.c_void_p
    lib.bartorch_noir_create.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.bartorch_noir_model.restype = ctypes.c_void_p
    lib.bartorch_noir_model.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_coils.restype = ctypes.c_void_p
    lib.bartorch_noir_coils.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_image.restype = ctypes.c_void_p
    lib.bartorch_noir_image.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_data.restype = ctypes.c_void_p
    lib.bartorch_noir_data.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_transform.restype = ctypes.c_void_p
    lib.bartorch_noir_transform.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_dims.restype = ctypes.c_int
    lib.bartorch_noir_dims.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
    ]
    lib.bartorch_noir_free.restype = None
    lib.bartorch_noir_free.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_net_create.restype = ctypes.c_void_p
    lib.bartorch_noir_net_create.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.bartorch_noir_net_step.restype = ctypes.c_void_p
    lib.bartorch_noir_net_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.bartorch_noir_net_iterations.restype = ctypes.c_void_p
    lib.bartorch_noir_net_iterations.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.bartorch_noir_net_adjoint.restype = ctypes.c_void_p
    lib.bartorch_noir_net_adjoint.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_net_decompose.restype = ctypes.c_void_p
    lib.bartorch_noir_net_decompose.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_net_split.restype = ctypes.c_void_p
    lib.bartorch_noir_net_split.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_net_join.restype = ctypes.c_void_p
    lib.bartorch_noir_net_join.argtypes = [ctypes.c_void_p]
    lib.bartorch_noir_net_free.restype = None
    lib.bartorch_noir_net_free.argtypes = [ctypes.c_void_p]
    lib.bartorch_irgnm.restype = ctypes.c_int
    lib.bartorch_irgnm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.bartorch_irgnm2.restype = ctypes.c_int
    lib.bartorch_irgnm2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.bartorch_cuda_built.restype = ctypes.c_int
    lib.bartorch_cuda_built.argtypes = []
    lib.bartorch_cuda_device_count.restype = ctypes.c_int
    lib.bartorch_cuda_device_count.argtypes = []
    lib.bartorch_cuda_enable.restype = ctypes.c_int
    lib.bartorch_cuda_enable.argtypes = [ctypes.c_int]
    lib.bartorch_cuda_device.restype = ctypes.c_int
    lib.bartorch_cuda_device.argtypes = []
    lib.bartorch_cuda_set_streams.restype = ctypes.c_int
    lib.bartorch_cuda_set_streams.argtypes = [ctypes.c_int]
    lib.bartorch_cuda_get_streams.restype = ctypes.c_int
    lib.bartorch_cuda_get_streams.argtypes = []
    lib.bartorch_cuda_use_memcache.restype = ctypes.c_int
    lib.bartorch_cuda_use_memcache.argtypes = [ctypes.c_int]
    lib.bartorch_cuda_memcache_clear_all.restype = None
    lib.bartorch_cuda_memcache_clear_all.argtypes = []
    lib.bartorch_cuda_wait_for_stream.restype = ctypes.c_int
    lib.bartorch_cuda_wait_for_stream.argtypes = [ctypes.c_void_p]
    lib.bartorch_cuda_signal_stream.restype = ctypes.c_int
    lib.bartorch_cuda_signal_stream.argtypes = [ctypes.c_void_p]
    lib.bartorch_cuda_free_memory.restype = ctypes.c_long
    lib.bartorch_cuda_free_memory.argtypes = []
    lib.bartorch_finufft_set.restype = ctypes.c_int
    lib.bartorch_finufft_set.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    lib.bartorch_finufft_layout.restype = ctypes.c_int
    lib.bartorch_finufft_layout.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.bartorch_finufft_set_tolerance.restype = None
    lib.bartorch_finufft_set_tolerance.argtypes = [ctypes.c_double]
    lib.bartorch_finufft_tolerance.restype = ctypes.c_double
    lib.bartorch_finufft_tolerance.argtypes = []
    lib.bartorch_finufft_set_upsampling.restype = None
    lib.bartorch_finufft_set_upsampling.argtypes = [ctypes.c_double]
    lib.bartorch_finufft_upsampling.restype = ctypes.c_double
    lib.bartorch_finufft_upsampling.argtypes = []
    lib.bartorch_finufft_set_threads.restype = None
    lib.bartorch_finufft_set_threads.argtypes = [ctypes.c_int]
    lib.bartorch_finufft_threads.restype = ctypes.c_int
    lib.bartorch_finufft_threads.argtypes = []
    lib.bartorch_finufft_use_in_tools.restype = None
    lib.bartorch_finufft_use_in_tools.argtypes = [ctypes.c_int]
    lib.bartorch_finufft_usable_on.restype = ctypes.c_int
    lib.bartorch_finufft_usable_on.argtypes = [ctypes.c_int]
    lib.bartorch_finufft_usable.restype = ctypes.c_int
    lib.bartorch_finufft_usable.argtypes = []
    lib.bartorch_finufft_live_plans.restype = ctypes.c_long
    lib.bartorch_finufft_live_plans.argtypes = []
    lib.bartorch_last_error.restype = ctypes.c_char_p
    lib.bartorch_last_error.argtypes = []
    lib.bartorch_clear_error.restype = None
    lib.bartorch_clear_error.argtypes = []
    lib.bartorch_nufft_decline_reason.restype = ctypes.c_int
    lib.bartorch_nufft_decline_reason.argtypes = []
    lib.bartorch_nufft_decline_text.restype = ctypes.c_char_p
    lib.bartorch_nufft_decline_text.argtypes = []
    lib.bartorch_nufft_allow_fallback.restype = None
    lib.bartorch_nufft_allow_fallback.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_fallback_allowed.restype = ctypes.c_int
    lib.bartorch_nufft_fallback_allowed.argtypes = []
    lib.bartorch_nufft_counter.restype = ctypes.c_long
    lib.bartorch_nufft_counter.argtypes = [ctypes.c_int]
    lib.bartorch_nufft_reset_counters.restype = None
    lib.bartorch_nufft_reset_counters.argtypes = []
    lib.bartorch_toeplitz_counter.restype = ctypes.c_long
    lib.bartorch_toeplitz_counter.argtypes = [ctypes.c_int]
    lib.bartorch_toeplitz_reset_counters.restype = None
    lib.bartorch_toeplitz_reset_counters.argtypes = []
    lib.bartorch_on_device.restype = ctypes.c_int
    lib.bartorch_on_device.argtypes = [ctypes.c_void_p]
    return lib
