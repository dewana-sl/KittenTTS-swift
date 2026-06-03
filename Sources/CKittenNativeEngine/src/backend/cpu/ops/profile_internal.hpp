#pragma once
#include <chrono>

static inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

extern double g_im2col_1x1_ms;
extern double g_im2col_3x3_ms;
extern double g_im2col_gen_ms;
extern double g_gemm_ms;
extern double g_add_req_ms;
extern double g_maxpool_ms;
extern double g_wino_transform_ms;
extern double g_wino_gemm_ms;
extern double g_wino_output_ms;

// ViT timers
extern double g_vit_ln_ms;
extern double g_vit_gelu_ms;
extern double g_vit_seqgemm_ms;
extern double g_vit_attn_ms;

// Depthwise timer
extern double g_depthwise_ms;

// LM decode timers
extern double g_lm_matvec_ms;   // seqgemm_int8_matvec total
extern double g_lm_attn_ms;     // causal_attn_core total
extern double g_lm_rmsnorm_ms;  // RMSNorm total

// TTS generator timers
extern double g_conv1d_fp32_ms;
extern double g_conv1d_int8_ms;
extern double g_ada_in1d_ms;
extern double g_snake1d_ms;
extern double g_layernorm_ms;
extern double g_conv_transpose_ms;
extern double g_sgemm_total_ms;
