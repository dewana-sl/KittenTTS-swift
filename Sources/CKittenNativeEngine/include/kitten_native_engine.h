#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* KTNativeModelHandle;

typedef struct KTNativeFloatArray {
    float* data;
    int length;
} KTNativeFloatArray;

KTNativeModelHandle kt_native_model_create(const char* arch_path,
                                           const char* weights_path,
                                           char** error_message);

void kt_native_model_destroy(KTNativeModelHandle handle);

int kt_native_model_synthesize(KTNativeModelHandle handle,
                               const float* phoneme_ids,
                               int phoneme_id_count,
                               const float* style,
                               int style_count,
                               KTNativeFloatArray* output,
                               char** error_message);

void kt_native_float_array_free(KTNativeFloatArray* array);

void kt_native_string_free(char* string);

#ifdef __cplusplus
}
#endif
