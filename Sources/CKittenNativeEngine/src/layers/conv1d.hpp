#pragma once
#include "engine/layer.hpp"

class Conv1dFp32 : public ILayer {
public:
    int C_out = 0, C_in = 0, kernel_size = 1;
    int stride = 1, padding = 0, dilation = 1, groups = 1;

    std::vector<float> weight_fp32;
    std::vector<float> bias_fp32;
    Buffer w_packed_f32;
    std::vector<Buffer> w_per_k_packed_;
    std::vector<Buffer> w_wino_packed_;
    Buffer w_kmerged_packed_;
    mutable std::vector<float> pad_buf_;
    mutable std::vector<float> tmp_buf_;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    MemSpace required_mem_space() const override {
        return active_backend()->mem_space;
    }
    Shape output_shape(Shape in) const override {
        int oT = (in.d1 + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;
        return {C_out, oT, 1};
    }
    void scratch_needed(Shape in, size_t out[Scratch::N]) const override {
        int oT = output_shape(in).d1;
        int C_in_g = C_in / std::max(groups, 1);
        size_t orig = (size_t)oT * kernel_size * C_in_g;
        size_t needed = orig;
        if (active_backend()->mem_space != MemSpace::CUDA &&
            active_backend()->mem_space != MemSpace::HIP) {
            if (groups == 1 && stride == 1) {
                int pad_rows = in.d1 + 2 * padding;
                if (kernel_size == 3 && dilation == 1) {
                    int n_tiles = (oT + 3) / 4;
                    int pad_ext = std::max(pad_rows, 4 * n_tiles + 2);
                    needed = std::max(orig, (size_t)pad_ext * C_in_g
                        + (size_t)6 * n_tiles * (C_in_g + C_out));
                } else if (kernel_size == 7 && dilation == 1) {
                    int n_tiles = (oT + 1) / 2;
                    int pad_ext = std::max(pad_rows, 2 * n_tiles + 6);
                    needed = std::max(orig, (size_t)pad_ext * C_in_g
                        + (size_t)8 * n_tiles * (C_in_g + C_out));
                } else if (kernel_size == 11 && dilation == 1) {
                    int n_tiles = (oT + 1) / 2;
                    int pad_ext = std::max(pad_rows, 2 * n_tiles + 10);
                    needed = std::max(orig, (size_t)pad_ext * C_in_g
                        + (size_t)12 * n_tiles * (C_in_g + C_out));
                } else {
                    needed = std::max(orig, (size_t)(in.d1 + 2 * padding) * C_in_g);
                }
            }
        }
        out[Scratch::F32_A] = std::max(out[Scratch::F32_A], needed * sizeof(float));
    }
};

class Conv1dInt8 : public ILayer {
public:
    int C_out = 0, C_in = 0, kernel_size = 1;
    int stride = 1, padding = 0, dilation = 1, groups = 1;

    std::vector<float>   w_scales;
    std::vector<float>   bias_fp32;
    std::vector<int64_t> eff_zeros;

    std::vector<Buffer> w_per_k_packed_;
    Buffer w_packed_full_;
    mutable std::vector<float>   im2col_buf_;

    Buffer w_gpu_packed_;

    mutable std::vector<float>   pad_buf_;
    mutable std::vector<float>   tmp_buf_;
    mutable std::vector<int8_t>  inp_scratch_;
    mutable std::vector<float>   req_scratch_;
    mutable std::vector<float>   row_scale_buf_;

    void     load_weights(const ReadTensorFn&) override;
    void     upload_weights(Allocator&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    MemSpace required_mem_space() const override {
        return active_backend()->mem_space;
    }
    Shape output_shape(Shape in) const override {
        int oT = (in.d1 + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;
        return {C_out, oT, 1};
    }
};

class ConvTranspose1dFp32 : public ILayer {
public:
    int C_out = 0, C_in = 0, kernel_size = 1;
    int stride = 1, padding = 0, output_padding = 0, groups = 1;

    std::vector<float> weight_fp32;
    std::vector<float> bias_fp32;
    std::vector<Buffer> w_packed_k;
    mutable std::vector<float> pad_buf_;
    std::vector<float> w_gpu_data;

    void     load_weights(const ReadTensorFn&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    MemSpace required_mem_space() const override {
        MemSpace ms = active_backend()->mem_space;
        return (ms == MemSpace::CUDA || ms == MemSpace::HIP) ? MemSpace::Host : ms;
    }
    Shape output_shape(Shape in) const override {
        int oT = (in.d1 - 1)*stride + kernel_size - 2*padding + output_padding;
        return {C_out, oT, 1};
    }
};
