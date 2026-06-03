#include "ffi/kitten_native_engine.h"

#include "engine/model.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" void kt_native_force_link_cpu_backend(void);

namespace {

struct NativeModel {
    explicit NativeModel(const char* arch_path, const char* weights_path)
        : model(arch_path, weights_path) {}

    InferenceModel model;
};

char* copy_string(const std::string& value) {
    char* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

void set_error(char** error_message, const std::string& message) {
    if (!error_message) return;
    *error_message = copy_string(message);
}

} // namespace

extern "C" {

KTNativeModelHandle kt_native_model_create(const char* arch_path,
                                           const char* weights_path,
                                           char** error_message) {
    kt_native_force_link_cpu_backend();

    if (error_message) *error_message = nullptr;
    if (!arch_path || !weights_path) {
        set_error(error_message, "arch_path and weights_path are required");
        return nullptr;
    }

    try {
        return new NativeModel(arch_path, weights_path);
    } catch (const std::exception& e) {
        set_error(error_message, e.what());
    } catch (...) {
        set_error(error_message, "unknown native engine create error");
    }
    return nullptr;
}

void kt_native_model_destroy(KTNativeModelHandle handle) {
    delete static_cast<NativeModel*>(handle);
}

int kt_native_model_synthesize(KTNativeModelHandle handle,
                               const float* phoneme_ids,
                               int phoneme_id_count,
                               const float* style,
                               int style_count,
                               KTNativeFloatArray* output,
                               char** error_message) {
    if (error_message) *error_message = nullptr;
    if (output) {
        output->data = nullptr;
        output->length = 0;
    }
    if (!handle || !phoneme_ids || phoneme_id_count <= 0 || !style || style_count < 256 || !output) {
        set_error(error_message, "invalid native synthesize arguments");
        return 0;
    }

    try {
        auto* native = static_cast<NativeModel*>(handle);

        std::vector<float> style_dec(128);
        std::vector<float> style_pred(128);
        std::copy(style, style + 128, style_dec.begin());
        std::copy(style + 128, style + 256, style_pred.begin());

        std::unordered_map<std::string, ModelIOTensor> inputs;
        inputs["phoneme_ids"] = ModelIOTensor(phoneme_ids, {phoneme_id_count, 1});
        inputs["style_dec"] = ModelIOTensor(style_dec.data(), {128});
        inputs["style_pred"] = ModelIOTensor(style_pred.data(), {128});

        auto outputs = native->model.forward(inputs);
        const auto& output_names = native->model.output_names();
        if (output_names.empty()) {
            set_error(error_message, "native model has no outputs");
            return 0;
        }

        auto it = outputs.find(output_names[0]);
        if (it == outputs.end()) {
            it = outputs.begin();
        }
        if (it == outputs.end()) {
            set_error(error_message, "native model returned no outputs");
            return 0;
        }

        const ModelIOTensor& tensor = it->second;
        const int count = static_cast<int>(tensor.numel());
        if (count <= 0) {
            set_error(error_message, "native model returned empty audio");
            return 0;
        }

        float* data = static_cast<float*>(std::malloc(sizeof(float) * static_cast<size_t>(count)));
        if (!data) {
            set_error(error_message, "failed to allocate native audio output");
            return 0;
        }
        std::memcpy(data, tensor.as_float(), sizeof(float) * static_cast<size_t>(count));

        output->data = data;
        output->length = count;
        return 1;
    } catch (const std::exception& e) {
        set_error(error_message, e.what());
    } catch (...) {
        set_error(error_message, "unknown native synthesize error");
    }
    return 0;
}

void kt_native_float_array_free(KTNativeFloatArray* array) {
    if (!array) return;
    std::free(array->data);
    array->data = nullptr;
    array->length = 0;
}

void kt_native_string_free(char* string) {
    std::free(string);
}

} // extern "C"
