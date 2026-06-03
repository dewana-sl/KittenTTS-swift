#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#include <cstdio>

double g_im2col_1x1_ms  = 0.0;
double g_im2col_3x3_ms  = 0.0;
double g_im2col_gen_ms  = 0.0;
double g_gemm_ms        = 0.0;
double g_add_req_ms     = 0.0;
double g_maxpool_ms     = 0.0;
double g_wino_transform_ms = 0.0;
double g_wino_gemm_ms      = 0.0;
double g_wino_output_ms    = 0.0;

double g_vit_ln_ms      = 0.0;
double g_vit_gelu_ms    = 0.0;
double g_vit_seqgemm_ms = 0.0;
double g_vit_attn_ms    = 0.0;

double g_depthwise_ms       = 0.0;

double g_lm_matvec_ms       = 0.0;
double g_lm_attn_ms         = 0.0;
double g_lm_rmsnorm_ms      = 0.0;

double g_conv1d_fp32_ms     = 0.0;
double g_conv1d_int8_ms     = 0.0;
double g_ada_in1d_ms        = 0.0;
double g_snake1d_ms         = 0.0;
double g_layernorm_ms       = 0.0;
double g_conv_transpose_ms  = 0.0;
double g_sgemm_total_ms     = 0.0;

void ops_profile_reset() {
    g_im2col_1x1_ms = g_im2col_3x3_ms = g_im2col_gen_ms = g_gemm_ms = g_add_req_ms = g_maxpool_ms = 0.0;
    g_wino_transform_ms = g_wino_gemm_ms = g_wino_output_ms = 0.0;
    g_vit_ln_ms = g_vit_gelu_ms = g_vit_seqgemm_ms = g_vit_attn_ms = 0.0;
    g_depthwise_ms = 0.0;
    g_conv1d_fp32_ms = g_conv1d_int8_ms = g_ada_in1d_ms = g_snake1d_ms = 0.0;
    g_layernorm_ms = g_conv_transpose_ms = g_sgemm_total_ms = 0.0;
    g_lm_matvec_ms = g_lm_attn_ms = g_lm_rmsnorm_ms = 0.0;
}

void ops_profile_print(int runs) {
    double d = (runs > 0) ? (double)runs : 1.0;
    double im2col_total = g_im2col_1x1_ms + g_im2col_3x3_ms + g_im2col_gen_ms;
    double vit_total    = g_vit_ln_ms + g_vit_gelu_ms + g_vit_seqgemm_ms + g_vit_attn_ms;
    printf("\n=== Op Profiling (avg per image, %d runs) ===\n", runs);
    printf("  im2col 1x1   : %6.3f ms\n", g_im2col_1x1_ms / d);
    printf("  im2col 3x3   : %6.3f ms\n", g_im2col_3x3_ms / d);
    printf("  im2col gen   : %6.3f ms\n", g_im2col_gen_ms / d);
    printf("  im2col total : %6.3f ms\n", im2col_total     / d);
    printf("  gemm         : %6.3f ms\n", g_gemm_ms        / d);
    printf("  wino xform   : %6.3f ms\n", g_wino_transform_ms / d);
    printf("  wino gemm    : %6.3f ms\n", g_wino_gemm_ms / d);
    printf("  wino output  : %6.3f ms\n", g_wino_output_ms / d);
    printf("  add_requant  : %6.3f ms\n", g_add_req_ms     / d);
    printf("  maxpool      : %6.3f ms\n", g_maxpool_ms     / d);
    if (vit_total > 0.0) {
        printf("  vit layernorm: %6.3f ms\n", g_vit_ln_ms      / d);
        printf("  vit gelu     : %6.3f ms\n", g_vit_gelu_ms    / d);
        printf("  vit seqgemm  : %6.3f ms\n", g_vit_seqgemm_ms / d);
        printf("  vit attention: %6.3f ms\n", g_vit_attn_ms    / d);
        printf("  vit total    : %6.3f ms\n", vit_total        / d);
    }
    printf("  depthwise    : %6.3f ms\n", g_depthwise_ms      / d);
    printf("  conv1d fp32  : %6.3f ms\n", g_conv1d_fp32_ms    / d);
    printf("  conv1d int8  : %6.3f ms\n", g_conv1d_int8_ms    / d);
    printf("  ada_in1d     : %6.3f ms\n", g_ada_in1d_ms       / d);
    printf("  snake1d      : %6.3f ms\n", g_snake1d_ms        / d);
    printf("  layernorm    : %6.3f ms\n", g_layernorm_ms      / d);
    printf("  conv_transp  : %6.3f ms\n", g_conv_transpose_ms / d);
    printf("  sgemm total  : %6.3f ms\n", g_sgemm_total_ms    / d);
    printf("  other (FC,...): [total - above]\n");
    printf("=============================================\n");
}
