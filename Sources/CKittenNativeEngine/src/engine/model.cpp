#include "model.hpp"
#include "arch_loader.hpp"
#include "weights_loader.hpp"
#include "tensor_pool.hpp"
#include "layers/factory.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <chrono>

static inline double _now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// ── Shape helpers ──────────────────────────────────────────────
// Internal layout: Shape{d0=C, d1=H_or_T, d2=W} (NHWC-derived).
// External format: compact vector — last element is C, leading elements are spatial dims.
// spec_to_shape  : compact vector[int]  →  Shape  (with optional runtime override)
// shape_to_compact: Shape            →  compact vector[int]  (inverse)

static Shape spec_to_shape(const std::vector<int>& spec,
                             const std::vector<int>& rt = {})
{
    int spec_size = (int)spec.size();
    int features  = spec_size >= 1 ? spec.back() : 1;  // features  → d0
    int primary   = spec_size >= 2 ? spec[0]     : 1;  // primary   → d1
    int secondary = spec_size >= 3 ? spec[1]     : 1;  // secondary → d2
    if (!rt.empty()) {
        int rt_size = (int)rt.size();
        if (features  == 0) features  = rt_size >= 1 ? rt.back() : 1;
        if (primary   == 0) primary   = rt_size >= 2 ? rt[0]     : 1;
        if (secondary == 0) secondary = rt_size >= 3 ? rt[1]     : 1;
    }
    return {features, std::max(primary, 1), std::max(secondary, 1)};
}

// Inverse of spec_to_shape: drops trailing d2=1 then d1=1.
static std::vector<int> shape_to_compact(const Shape& b) {
    if (b.d2 > 1) return {b.d1, b.d2, b.d0};
    if (b.d1 > 1) return {b.d1, b.d0};
    return {b.d0};
}

// Shared CPU allocator for host-side staging buffers.
static CpuAllocator s_cpu_alloc;


// ──────────────────────────────────────────────────────────────
// update_buffers — propagate shapes + grow-only buffer allocation.
//
// For each input in loaded_arch_.arch.all_inputs:
//   - Fixed dims (> 0 in InputSpec) are taken from the spec.
//   - Variable dims (== 0 in InputSpec) are filled from the matching
//     runtime ModelIOTensor's shape (if present in `inputs`); default to 1.
// Shapes are then propagated through the graph and buffers / scratch
// are grown as needed.  Called from the constructor (with an empty map,
// which uses spec sizes for all fixed-size models) and at the start of
// every forward() call.
// ──────────────────────────────────────────────────────────────
void InferenceModel::update_buffers(
    const std::unordered_map<std::string, ModelIOTensor>& inputs)
{
    // ── Resolve concrete Shape for every input ─────────────
    std::unordered_map<std::string, Shape> input_shapes;
    for (const auto& inp : loaded_arch_.arch.all_inputs) {
        auto it = inputs.find(inp.name);
        input_shapes[inp.name] = spec_to_shape(
            inp.shape,
            it != inputs.end() ? it->second.shape : std::vector<int>{});
    }

    auto result = pool_.propagate_shapes(loaded_arch_.arch, input_shapes);

    // ── Grow-only buffer allocation ───────────────────────────
    for (int slot = 0; slot < pool_.slots.num_slots; ++slot)
        pool_.grow(slot, result.slot_sizes[slot], *alloc_);

    // ── Grow-only scratch allocation ──────────────────────────
    for (int node_idx = 0; node_idx < (int)loaded_arch_.arch.nodes.size(); ++node_idx) {
        if (!pool_.node_layers[node_idx]) continue;
        const std::string& in0 = loaded_arch_.arch.nodes[node_idx].inputs.empty()
            ? std::string("") : loaded_arch_.arch.nodes[node_idx].inputs[0];
        Shape input_shape = result.shapes.count(in0) ? result.shapes.at(in0) : Shape{};
        scratch_.grow_for_layer(pool_.node_layers[node_idx], input_shape, *alloc_);
    }
}


// ──────────────────────────────────────────────────────────────
// dispatch_layer — run one layer, staging inputs/output to host if needed.
// `inputs` is a mutable stack array — views are replaced with host copies
// in the staging path without affecting the caller's slot buffers.
// When inputs_staged=true the caller has already copied inputs into
// h_stage_ins and updated inputs[i].view; only output staging is performed.
// ──────────────────────────────────────────────────────────────
static Shape dispatch_layer(
    ILayer* layer,
    TensorView* inputs, int n_inputs,
    BufferView out_view,
    ScratchPads& scratch,
    Allocator* alloc,
    Buffer* h_stage_ins,   // [MAX_LAYER_INPUTS], grow-only host buffers
    Buffer& h_stage_out,
    bool inputs_staged = false)
{
    const bool needs_staging = (layer->required_mem_space() != active_backend()->mem_space);
    const size_t out_device_bytes = out_view.bytes;

    if (needs_staging) {
        auto grow = [](Buffer& buf, size_t need) {
            if (!buf.valid() || buf.bytes() < need) buf = s_cpu_alloc.make_buffer(need);
        };
        void* dev_out = out_view.ptr;
        if (!inputs_staged) {
            for (int i = 0; i < n_inputs; ++i) {
                grow(h_stage_ins[i], inputs[i].view.bytes);
                alloc->copy_d2h(h_stage_ins[i].ptr(), inputs[i].view.ptr, inputs[i].view.bytes);
                inputs[i].view = h_stage_ins[i].view;
            }
        }
        grow(h_stage_out, out_view.bytes);
        out_view = h_stage_out.view;
        Shape output_shape = layer->forward(inputs, n_inputs, out_view, scratch);
        // Use out_device_bytes (original device buffer size), not out_view.bytes which
        // may be larger if the staging buffer was reused from a previous larger layer.
        alloc->copy_h2d(dev_out, h_stage_out.ptr(), out_device_bytes);
        return output_shape;
    }
    return layer->forward(inputs, n_inputs, out_view, scratch);
}

// ──────────────────────────────────────────────────────────────
// execute_graph — run all graph nodes using pre-built ScratchPads
// ──────────────────────────────────────────────────────────────
void InferenceModel::execute_graph(ScratchPads& scratch) {
    for (int node_idx = 0; node_idx < (int)loaded_arch_.arch.nodes.size(); ++node_idx) {
        const GraphNode& node = loaded_arch_.arch.nodes[node_idx];
        ILayer* layer = pool_.node_layers[node_idx];
        if (!layer) throw std::runtime_error("No layer for: " + node.name);
        const std::string& out_name = node.outputs.empty() ? std::string("") : node.outputs[0];
        int out_slot = pool_.slots.slot_map.at(out_name);

        // Resolve slot index for each input
        int n_inputs = (int)node.inputs.size();
        if (n_inputs > MAX_LAYER_INPUTS)
            throw std::runtime_error("node '" + node.name + "' has " +
                                     std::to_string(n_inputs) + " inputs (max " +
                                     std::to_string(MAX_LAYER_INPUTS) + ")");
        int in_slots[MAX_LAYER_INPUTS] = {};
        for (int i = 0; i < n_inputs; ++i)
            in_slots[i] = pool_.slots.slot_map.at(node.inputs[i]);

        // If the layer requires host memory (CPU-only under a device backend), stage all
        // inputs into h_stage_ins_ once.  The same staged copies are reused for both
        // dynamic_output_size (which may read tensor contents) and forward().
        const bool needs_staging = (layer->required_mem_space() != active_backend()->mem_space);
        const void* dyn_ptrs[MAX_LAYER_INPUTS] = {};
        Shape    dyn_shapes[MAX_LAYER_INPUTS] = {};
        for (int i = 0; i < n_inputs; ++i) {
            dyn_ptrs[i]   = pool_.raw_ptr(in_slots[i]);
            dyn_shapes[i] = pool_.slot_shapes[in_slots[i]];
        }
        if (needs_staging) {
            for (int i = 0; i < n_inputs; ++i) {
                size_t size_bytes = pool_.buffers[in_slots[i]].bytes();
                if (size_bytes > 0) {
                    if (!h_stage_ins_[i].valid() || h_stage_ins_[i].bytes() < size_bytes)
                        h_stage_ins_[i] = s_cpu_alloc.make_buffer(size_bytes);
                    alloc_->copy_d2h(h_stage_ins_[i].ptr(), dyn_ptrs[i], size_bytes);
                    dyn_ptrs[i] = h_stage_ins_[i].ptr();
                }
            }
        }

        // Grow output buffer to fit dynamic output shape
        Shape output_shape = layer->dynamic_output_size(dyn_ptrs, dyn_shapes, n_inputs);
        {
            size_t need_bytes = (size_t)output_shape.d0 * (size_t)output_shape.d1 * (size_t)output_shape.d2
                                * pool_.slot_dtype[out_slot].elem_bytes();
            if (need_bytes > pool_.buffers[out_slot].bytes())
                pool_.buffers[out_slot] = alloc_->make_buffer(need_bytes);
        }

        // Grow scratch buffers for this layer.
        // This intentionally runs per-node rather than just once in update_buffers:
        // variable-length outputs (e.g. LengthRegulate) can change the input shape
        // seen by later layers at runtime, growing scratch requirements beyond the
        // static estimate computed during update_buffers.
        {
            size_t sizes[Scratch::N] = {};
            layer->scratch_needed(n_inputs > 0 ? pool_.slot_shapes[in_slots[0]] : Shape{}, sizes);
            for (int i = 0; i < Scratch::N; ++i) {
                scratch_.grow(i, sizes[i], *alloc_);
                scratch.buffers[i] = scratch_.buffers[i].view;
            }
        }

        // Build TensorView array from slot buffers.
        // If inputs were staged above, use the staged host views so dispatch_layer
        // can skip the redundant d2h copy.
        TensorView inp_buf[MAX_LAYER_INPUTS];
        for (int i = 0; i < n_inputs; ++i) {
            inp_buf[i].view  = (needs_staging && h_stage_ins_[i].valid())
                               ? h_stage_ins_[i].view : pool_.buffers[in_slots[i]].view;
            inp_buf[i].shape = pool_.slot_shapes[in_slots[i]];
        }
        BufferView out_view = pool_.buffers[out_slot].view;

        double start_ms = _now_ms();
        pool_.slot_shapes[out_slot] = dispatch_layer(layer, inp_buf, n_inputs, out_view, scratch,
                                                     alloc_, h_stage_ins_, h_stage_out_,
                                                     needs_staging);
        { double elapsed_ms = _now_ms() - start_ms; op_ms_[op_type_name(node.op)] += elapsed_ms; layer_ms_[node.name] += elapsed_ms; }
    }
}

// ──────────────────────────────────────────────────────────────
// Constructor
// ──────────────────────────────────────────────────────────────
InferenceModel::InferenceModel(const std::string& arch_path,
                           const std::string& weights_path)
{
    // Cache the backend's allocator singleton once.  Both CPU and CUDA backends
    // return a pointer to a static object, so this pointer remains valid for the
    // lifetime of the process.  All Buffer deleters capture this pointer,
    // which is safe because the allocator outlives every buffer we create.
    alloc_ = active_backend()->make_allocator();

    loaded_arch_ = ArchLoader::load(arch_path);
    load_weights(weights_path, loaded_arch_.layers);
    pool_ = BufferPool::build(GraphSlots::build(loaded_arch_.arch),
                              loaded_arch_.arch, loaded_arch_.layers);
    update_buffers({});   // initial sizing: uses InputSpec shapes (variable dims → 1)

    for (auto& layer : loaded_arch_.layers)
        if (layer) layer->upload_weights(*alloc_);
}

// ──────────────────────────────────────────────────────────────
// GPU-aware buffer helpers
// ──────────────────────────────────────────────────────────────
static void buf_write(Allocator* alloc, void* dst, const void* src, size_t bytes) {
    if (alloc->mem_space() == MemSpace::Host)
        ::memcpy(dst, src, bytes);
    else
        alloc->copy_h2d(dst, src, bytes);
}
static void buf_read(Allocator* alloc, void* dst, const void* src, size_t bytes) {
    if (alloc->mem_space() == MemSpace::Host)
        ::memcpy(dst, src, bytes);
    else
        alloc->copy_d2h(dst, src, bytes);
}

// ──────────────────────────────────────────────────────────────
// forward — dict of named NHWC input IOTensors → dict of output IOTensors
// ──────────────────────────────────────────────────────────────
std::unordered_map<std::string, ModelIOTensor>
InferenceModel::forward(const std::unordered_map<std::string, ModelIOTensor>& inputs,
                        int step_pos, bool is_decode)
{
    // ── Grow buffers to fit these inputs (no-op if already large enough) ──
    update_buffers(inputs);

    // ── Copy inputs into device/host buffers ──────────────────
    for (const auto& inp : loaded_arch_.arch.all_inputs) {
        auto it = inputs.find(inp.name);
        if (it == inputs.end()) continue;
        const ModelIOTensor& tensor = it->second;
        int slot = pool_.slots.slot_map.at(inp.name);

        // NHWC compact: last dim = features (d0), first = primary (d1), second = secondary (d2)
        const Shape shape = spec_to_shape(tensor.shape);
        buf_write(alloc_, pool_.raw_ptr(slot), tensor.data, tensor.nbytes());
        pool_.slot_shapes[slot] = shape;
    }

    ScratchPads scratch = scratch_.make_pads();
    scratch.step_pos  = step_pos;
    scratch.is_decode = is_decode;
    execute_graph(scratch);

    // ── Collect named outputs ─────────────────────────────────
    std::unordered_map<std::string, ModelIOTensor> result;
    for (const auto& oname : loaded_arch_.arch.output_names) {
        auto slot_it = pool_.slots.slot_map.find(oname);
        if (slot_it == pool_.slots.slot_map.end()) continue;
        int slot = slot_it->second;
        const Shape& output_shape = pool_.slot_shapes[slot];
        const size_t num_elems = (size_t)output_shape.d0 * (size_t)output_shape.d1 * (size_t)output_shape.d2;
        const size_t out_bytes = num_elems * pool_.slot_dtype[slot].elem_bytes();
        ModelIOTensor output_tensor;
        output_tensor.shape = shape_to_compact(output_shape);
        output_tensor.dtype = pool_.slot_dtype[slot].dtype_str();
        output_tensor.owned_.resize(out_bytes);
        buf_read(alloc_, output_tensor.owned_.data(), pool_.buffers[slot].ptr(), out_bytes);
        output_tensor.data = output_tensor.owned_.data();
        result[oname] = std::move(output_tensor);
    }
    return result;
}

// ──────────────────────────────────────────────────────────────
// reset_state — allocate / reset KV caches in all stateful layers
// ──────────────────────────────────────────────────────────────
void InferenceModel::reset_state(int max_seq_len) {
    for (auto& layer : loaded_arch_.layers)
        if (layer && layer->is_stateful())
            layer->reset_state(max_seq_len, *alloc_);
}

