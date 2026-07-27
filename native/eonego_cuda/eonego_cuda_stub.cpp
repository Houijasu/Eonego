#include "eonego_cuda.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

struct CudaBackend;

struct Tensor {
    std::string name;
    std::vector<uint32_t> dims;
    std::vector<float> data;
};

struct EonegoGpuHandle {
    std::string model_path;
    int device_id;
    int max_batch;
    uint32_t d_model;
    uint32_t layers;
    uint32_t value_scale;
    uint32_t tensor_count;
    std::vector<Tensor> tensors;
    std::unordered_map<std::string, size_t> tensor_index;
    bool has_graphnet_weights;
    bool infer_ready;
    CudaBackend* cuda;
};

static constexpr uint32_t EONGR_VERSION = 1;
static constexpr uint32_t EONGR_SCHEMA = 0x47463101u;
static constexpr uint32_t DTYPE_F32 = 1;
static constexpr int RECORD_BYTES = EOGPU_GRAPH_INPUT_BYTES;
static constexpr int OUTPUT_INTS = EOGPU_GRAPH_OUTPUT_BYTES / static_cast<int>(sizeof(int32_t));
static constexpr int NODE_COUNT = 32;
static constexpr int NODE_FEATURES = 12;
static constexpr int EDGE_FEATURES = 8;
static constexpr int NODE_OFFSET = 4;
static constexpr int EDGE_OFFSET = NODE_OFFSET + NODE_COUNT * NODE_FEATURES;
static constexpr int POLICY_HEAD = 384;

enum class BackendKind {
    CpuScalar,
    Cuda
};

static int clamp_i32(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void set_error(char* err_buf, int err_len, const char* msg) {
    if (err_buf == nullptr || err_len <= 0) return;
    std::snprintf(err_buf, static_cast<size_t>(err_len), "%s", msg);
}

#ifdef _WIN32
using CUdevice = int;
using CUresult = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = unsigned long long;
using nvrtcProgram = void*;
using nvrtcResult = int;

struct CudaApi {
    HMODULE driver = nullptr;
    HMODULE nvrtc = nullptr;
    int (*cuInit)(unsigned int) = nullptr;
    int (*cuDeviceGet)(CUdevice*, int) = nullptr;
    int (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = nullptr;
    int (*cuCtxSetCurrent)(CUcontext) = nullptr;
    int (*cuCtxDestroy)(CUcontext) = nullptr;
    int (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
    int (*cuModuleUnload)(CUmodule) = nullptr;
    int (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    int (*cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
    int (*cuMemFree)(CUdeviceptr) = nullptr;
    int (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    int (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    int (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) = nullptr;
    int (*cuCtxSynchronize)() = nullptr;

    nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*) = nullptr;
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
};

struct CudaBackend {
    CudaApi api;
    CUcontext ctx = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUfunction learned_kernel = nullptr;
    CUfunction learned_kernel128 = nullptr;
    CUdeviceptr d_weights = 0;
    CUdeviceptr d_offsets = 0;
    CUdeviceptr d_input = 0;
    CUdeviceptr d_output = 0;
    size_t weight_bytes = 0;
    size_t offset_bytes = 0;
    size_t input_capacity = 0;
    size_t output_capacity = 0;
    bool ready = false;
    bool learned_ready = false;
    std::string why;
    std::mutex mutex;
};

static FARPROC load_proc(HMODULE dll, const char* name) {
    return dll == nullptr ? nullptr : GetProcAddress(dll, name);
}

template <typename T>
static bool bind_proc(HMODULE dll, const char* name, T& out) {
    out = reinterpret_cast<T>(load_proc(dll, name));
    return out != nullptr;
}

static HMODULE load_nvrtc() {
    HMODULE h = LoadLibraryA("nvrtc64_120_0.dll");
    if (h != nullptr) return h;

    char cuda_path[MAX_PATH] = {};
    DWORD n = GetEnvironmentVariableA("CUDA_PATH", cuda_path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        n = GetEnvironmentVariableA("CUDA_HOME", cuda_path, MAX_PATH);
    }
    if (n == 0 || n >= MAX_PATH) return nullptr;

    std::string bin = std::string(cuda_path) + "\\bin";
    SetDllDirectoryA(bin.c_str());
    std::string dll = bin + "\\nvrtc64_120_0.dll";
    return LoadLibraryA(dll.c_str());
}

static bool load_cuda_api(CudaApi& a, std::string& why) {
    a.driver = LoadLibraryA("nvcuda.dll");
    if (a.driver == nullptr) {
        why = "nvcuda.dll not available";
        return false;
    }
    a.nvrtc = load_nvrtc();
    if (a.nvrtc == nullptr) {
        why = "nvrtc64_120_0.dll not available";
        return false;
    }

    bool ok =
        bind_proc(a.driver, "cuInit", a.cuInit)
        && bind_proc(a.driver, "cuDeviceGet", a.cuDeviceGet)
        && bind_proc(a.driver, "cuCtxCreate_v2", a.cuCtxCreate)
        && bind_proc(a.driver, "cuCtxSetCurrent", a.cuCtxSetCurrent)
        && bind_proc(a.driver, "cuCtxDestroy_v2", a.cuCtxDestroy)
        && bind_proc(a.driver, "cuModuleLoadData", a.cuModuleLoadData)
        && bind_proc(a.driver, "cuModuleUnload", a.cuModuleUnload)
        && bind_proc(a.driver, "cuModuleGetFunction", a.cuModuleGetFunction)
        && bind_proc(a.driver, "cuMemAlloc_v2", a.cuMemAlloc)
        && bind_proc(a.driver, "cuMemFree_v2", a.cuMemFree)
        && bind_proc(a.driver, "cuMemcpyHtoD_v2", a.cuMemcpyHtoD)
        && bind_proc(a.driver, "cuMemcpyDtoH_v2", a.cuMemcpyDtoH)
        && bind_proc(a.driver, "cuLaunchKernel", a.cuLaunchKernel)
        && bind_proc(a.driver, "cuCtxSynchronize", a.cuCtxSynchronize)
        && bind_proc(a.nvrtc, "nvrtcCreateProgram", a.nvrtcCreateProgram)
        && bind_proc(a.nvrtc, "nvrtcCompileProgram", a.nvrtcCompileProgram)
        && bind_proc(a.nvrtc, "nvrtcGetPTXSize", a.nvrtcGetPTXSize)
        && bind_proc(a.nvrtc, "nvrtcGetPTX", a.nvrtcGetPTX)
        && bind_proc(a.nvrtc, "nvrtcGetProgramLogSize", a.nvrtcGetProgramLogSize)
        && bind_proc(a.nvrtc, "nvrtcGetProgramLog", a.nvrtcGetProgramLog)
        && bind_proc(a.nvrtc, "nvrtcDestroyProgram", a.nvrtcDestroyProgram);

    if (!ok) {
        why = "CUDA driver/NVRTC entry point missing";
    }
    return ok;
}

static const char* cuda_kernel_src = R"CUDA(
extern "C" __device__ int eonego_clamp_i32(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
extern "C" __device__ int eonego_abs_i32(int v) { return v < 0 ? -v : v; }
extern "C" __device__ int eonego_piece_value(int pt) {
    return pt == 0 ? 100 : (pt == 1 ? 320 : (pt == 2 ? 330 : (pt == 3 ? 500 : (pt == 4 ? 900 : 0))));
}
extern "C" __device__ int eonego_center_bonus(int rel_sq) {
    int f = rel_sq & 7;
    int r = rel_sq >> 3;
    int df3 = eonego_abs_i32(f - 3), df4 = eonego_abs_i32(f - 4);
    int dr3 = eonego_abs_i32(r - 3), dr4 = eonego_abs_i32(r - 4);
    int df = df3 < df4 ? df3 : df4;
    int dr = dr3 < dr4 ? dr3 : dr4;
    return 18 - 4 * (df + dr);
}
extern "C" __device__ void eonego_write_wdl(int value_cp, int* out) {
    int abs_cp = eonego_abs_i32(value_cp);
    int draw = eonego_clamp_i32(360 - abs_cp / 2, 40, 620);
    int decisive = 1000 - draw;
    int win = decisive / 2 + eonego_clamp_i32(value_cp / 4, -decisive / 2, decisive / 2);
    win = eonego_clamp_i32(win, 0, 1000 - draw);
    out[1] = win;
    out[2] = draw;
    out[3] = 1000 - draw - win;
}
extern "C" __global__ void eonego_graph_infer(const unsigned char* input, int* output, int batch_count) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_count) return;
    const int RECORD_BYTES = 8612;
    const int OUTPUT_INTS = 772;
    const int NODE_COUNT = 32;
    const int NODE_FEATURES = 12;
    const int EDGE_FEATURES = 8;
    const int NODE_OFFSET = 4;
    const int EDGE_OFFSET = NODE_OFFSET + NODE_COUNT * NODE_FEATURES;
    const int POLICY_HEAD = 384;
    const unsigned char* in = input + b * RECORD_BYTES;
    int* out = output + b * OUTPUT_INTS;
    for (int i = 0; i < OUTPUT_INTS; ++i) out[i] = 0;
    int node_count = eonego_clamp_i32((int)in[0], 0, NODE_COUNT);
    bool in_check = in[2] != 0;
    int checker_count = (int)in[3];
    int score = 0, own_bishops = 0, opp_bishops = 0;
    for (int i = 0; i < node_count; ++i) {
        const unsigned char* n = in + NODE_OFFSET + i * NODE_FEATURES;
        if (n[0] == 0) continue;
        bool own = n[1] != 0;
        int pt = (int)n[3];
        int rel_sq = (int)n[5];
        int rel_rank = rel_sq >> 3;
        int v = eonego_piece_value(pt);
        v += eonego_center_bonus(rel_sq);
        if (pt == 0) v += rel_rank * 7;
        if (n[8] != 0) v -= 18;
        if (n[9] != 0) v += 10;
        if (n[10] != 0) v += 28;
        if (pt == 2) { if (own) ++own_bishops; else ++opp_bishops; }
        score += own ? v : -v;
        if (pt >= 0 && pt < 6) {
            int idx = pt * 64 + rel_sq;
            if (idx >= 0 && idx < POLICY_HEAD) out[4 + idx] += own ? 900 + v / 2 : -350;
        }
    }
    if (own_bishops >= 2) score += 35;
    if (opp_bishops >= 2) score -= 35;
    if (in_check) score -= 55 + 30 * checker_count;
    for (int i = 0; i < node_count; ++i) {
        const unsigned char* a = in + NODE_OFFSET + i * NODE_FEATURES;
        if (a[0] == 0 || a[1] == 0) continue;
        int apt = (int)a[3], arel = (int)a[5];
        if (apt < 0 || apt >= 6 || arel < 0 || arel >= 64) continue;
        for (int j = 0; j < node_count; ++j) {
            if (i == j) continue;
            const unsigned char* bb = in + NODE_OFFSET + j * NODE_FEATURES;
            if (bb[0] == 0) continue;
            const unsigned char* e = in + EDGE_OFFSET + ((i * NODE_COUNT) + j) * EDGE_FEATURES;
            if (e[0] == 0) continue;
            bool attacks = e[2] != 0;
            bool same_color = e[1] != 0;
            int bpt = (int)bb[3], brel = (int)bb[5];
            if (attacks && !same_color) {
                score += eonego_clamp_i32(eonego_piece_value(bpt) / 16, 4, 60);
                int to_idx = apt * 64 + brel;
                if (to_idx >= 0 && to_idx < POLICY_HEAD) out[4 + POLICY_HEAD + to_idx] += 600 + eonego_piece_value(bpt) / 2;
            } else if (attacks && same_color) {
                int to_idx = apt * 64 + brel;
                if (to_idx >= 0 && to_idx < POLICY_HEAD) out[4 + POLICY_HEAD + to_idx] += 35;
            }
        }
    }
    out[0] = eonego_clamp_i32(score, -2500, 2500);
    eonego_write_wdl(out[0], out);
}
extern "C" __device__ float eonego_linear_at(const float* w, const float* b, const float* x, int out_idx, int in_dim) {
    const float* row = w + out_idx * in_dim;
    float y = b[out_idx];
    for (int i = 0; i < in_dim; ++i) y += row[i] * x[i];
    return y;
}
extern "C" __device__ int eonego_scaled_logit(float x) {
    float y = roundf(x * 1024.0f);
    if (y > 2147483647.0f) return 2147483647;
    if (y < -2147483648.0f) return -2147483647 - 1;
    return (int)y;
}
extern "C" __device__ void eonego_wdl_softmax(float a, float b, float c, int* out) {
    float m = fmaxf(a, fmaxf(b, c));
    float ea = expf(a - m), eb = expf(b - m), ec = expf(c - m);
    float s = ea + eb + ec;
    int w = (int)roundf(1000.0f * ea / s);
    int d = (int)roundf(1000.0f * eb / s);
    int l = 1000 - w - d;
    if (l < 0) { l = 0; if (w >= d) --w; else --d; }
    out[1] = eonego_clamp_i32(w, 0, 1000);
    out[2] = eonego_clamp_i32(d, 0, 1000 - out[1]);
    out[3] = 1000 - out[1] - out[2];
}
extern "C" __device__ void eonego_ln32(float* x, const float* gamma, const float* beta) {
    float mean = 0.0f;
    for (int i = 0; i < 32; ++i) mean += x[i];
    mean *= 1.0f / 32.0f;
    float var = 0.0f;
    for (int i = 0; i < 32; ++i) { float z = x[i] - mean; var += z * z; }
    float inv = rsqrtf(var * (1.0f / 32.0f) + 1.0e-5f);
    for (int i = 0; i < 32; ++i) x[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
}
extern "C" __global__ void eonego_graphnet32_l1_infer(const unsigned char* input, const float* weights, const int* off, int* output, int batch_count, int value_scale) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_count) return;
    const int RECORD_BYTES = 8612;
    const int OUTPUT_INTS = 772;
    const int NODE_COUNT = 32;
    const int NODE_FEATURES = 12;
    const int EDGE_FEATURES = 8;
    const int NODE_OFFSET = 4;
    const int EDGE_OFFSET = NODE_OFFSET + NODE_COUNT * NODE_FEATURES;
    const int GLOBAL_OFFSET = EDGE_OFFSET + NODE_COUNT * NODE_COUNT * EDGE_FEATURES;
    const int POLICY_HEAD = 384;
    const unsigned char* in = input + b * RECORD_BYTES;
    int* out = output + b * OUTPUT_INTS;
    for (int i = 0; i < OUTPUT_INTS; ++i) out[i] = 0;
    int node_count = eonego_clamp_i32((int)in[0], 0, NODE_COUNT);

    const float* node_w = weights + off[0];  const float* node_b = weights + off[1];
    const float* edge_w = weights + off[2];  const float* edge_b = weights + off[3];
    const float* glob_w = weights + off[4];  const float* glob_b = weights + off[5];
    const float* msg_w = weights + off[6];   const float* msg_b = weights + off[7];
    const float* upd_w = weights + off[8];   const float* upd_b = weights + off[9];
    const float* norm_w = weights + off[10]; const float* norm_b = weights + off[11];
    const float* v0_w = weights + off[12];   const float* v0_b = weights + off[13];
    const float* v2_w = weights + off[14];   const float* v2_b = weights + off[15];
    const float* w0_w = weights + off[16];   const float* w0_b = weights + off[17];
    const float* w2_w = weights + off[18];   const float* w2_b = weights + off[19];
    const float* fh_w = weights + off[20];   const float* fh_b = weights + off[21];
    const float* th_w = weights + off[22];   const float* th_b = weights + off[23];

    float x[32 * 32];
    float nx[32 * 32];
    float g[32];
    float feat32[32];
    for (int n = 0; n < NODE_COUNT; ++n) {
        const unsigned char* src = in + NODE_OFFSET + n * NODE_FEATURES;
        for (int k = 0; k < NODE_FEATURES; ++k) feat32[k] = ((float)src[k]) * (1.0f / 255.0f);
        for (int o = 0; o < 32; ++o) {
            float y = node_b[o];
            for (int k = 0; k < NODE_FEATURES; ++k) y += node_w[o * NODE_FEATURES + k] * feat32[k];
            x[n * 32 + o] = n < node_count ? y : 0.0f;
        }
    }
    for (int k = 0; k < 32; ++k) feat32[k] = ((float)in[GLOBAL_OFFSET + k]) * (1.0f / 255.0f);
    for (int o = 0; o < 32; ++o) {
        float y = glob_b[o];
        for (int k = 0; k < 32; ++k) y += glob_w[o * 32 + k] * feat32[k];
        g[o] = y;
    }

    float accum[32], edgev[32], msg_in[96], upd_in[64], upd_out[32];
    for (int dst = 0; dst < node_count; ++dst) {
        for (int k = 0; k < 32; ++k) accum[k] = 0.0f;
        const float* x_dst = x + dst * 32;
        for (int src_i = 0; src_i < node_count; ++src_i) {
            const float* x_src = x + src_i * 32;
            const unsigned char* er = in + EDGE_OFFSET + ((src_i * NODE_COUNT) + dst) * EDGE_FEATURES;
            for (int k = 0; k < EDGE_FEATURES; ++k) feat32[k] = ((float)er[k]) * (1.0f / 255.0f);
            for (int o = 0; o < 32; ++o) {
                float y = edge_b[o];
                for (int k = 0; k < EDGE_FEATURES; ++k) y += edge_w[o * EDGE_FEATURES + k] * feat32[k];
                edgev[o] = y;
            }
            for (int k = 0; k < 32; ++k) {
                msg_in[k] = x_src[k];
                msg_in[32 + k] = x_dst[k];
                msg_in[64 + k] = edgev[k];
            }
            for (int o = 0; o < 32; ++o) {
                float y = eonego_linear_at(msg_w, msg_b, msg_in, o, 96);
                if (y > 0.0f) accum[o] += y;
            }
        }
        for (int k = 0; k < 32; ++k) { upd_in[k] = x_dst[k]; upd_in[32 + k] = accum[k]; }
        for (int o = 0; o < 32; ++o) {
            float y = eonego_linear_at(upd_w, upd_b, upd_in, o, 64);
            upd_out[o] = y > 0.0f ? y : 0.0f;
            nx[dst * 32 + o] = x_dst[o] + upd_out[o];
        }
        eonego_ln32(nx + dst * 32, norm_w, norm_b);
    }
    for (int n = node_count; n < NODE_COUNT; ++n) for (int k = 0; k < 32; ++k) nx[n * 32 + k] = 0.0f;

    float pooled[32];
    for (int k = 0; k < 32; ++k) pooled[k] = 0.0f;
    float denom = node_count > 0 ? (float)node_count : 1.0f;
    for (int n = 0; n < node_count; ++n) for (int k = 0; k < 32; ++k) pooled[k] += nx[n * 32 + k] / denom;

    float h[64], hidden[32];
    for (int k = 0; k < 32; ++k) { h[k] = pooled[k]; h[32 + k] = g[k]; }
    for (int o = 0; o < 32; ++o) { float y = eonego_linear_at(v0_w, v0_b, h, o, 64); hidden[o] = y > 0.0f ? y : 0.0f; }
    float value = eonego_linear_at(v2_w, v2_b, hidden, 0, 32);
    out[0] = eonego_clamp_i32((int)roundf(value * (float)value_scale), -2500, 2500);

    for (int o = 0; o < 32; ++o) { float y = eonego_linear_at(w0_w, w0_b, h, o, 64); hidden[o] = y > 0.0f ? y : 0.0f; }
    eonego_wdl_softmax(
        eonego_linear_at(w2_w, w2_b, hidden, 0, 32),
        eonego_linear_at(w2_w, w2_b, hidden, 1, 32),
        eonego_linear_at(w2_w, w2_b, hidden, 2, 32),
        out);
    for (int i = 0; i < POLICY_HEAD; ++i) {
        out[4 + i] = eonego_scaled_logit(eonego_linear_at(fh_w, fh_b, h, i, 64));
        out[4 + POLICY_HEAD + i] = eonego_scaled_logit(eonego_linear_at(th_w, th_b, h, i, 64));
    }
}
extern "C" __device__ void eonego_ln128(float* x, const float* gamma, const float* beta) {
    float mean = 0.0f;
    for (int i = 0; i < 128; ++i) mean += x[i];
    mean *= 1.0f / 128.0f;
    float var = 0.0f;
    for (int i = 0; i < 128; ++i) { float z = x[i] - mean; var += z * z; }
    float inv = rsqrtf(var * (1.0f / 128.0f) + 1.0e-5f);
    for (int i = 0; i < 128; ++i) x[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
}
extern "C" __global__ void eonego_graphnet128_l3_infer(const unsigned char* input, const float* weights, const int* off, int* output, int batch_count, int value_scale) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_count) return;
    const int RECORD_BYTES = 8612;
    const int OUTPUT_INTS = 772;
    const int NODE_COUNT = 32;
    const int NODE_FEATURES = 12;
    const int EDGE_FEATURES = 8;
    const int NODE_OFFSET = 4;
    const int EDGE_OFFSET = NODE_OFFSET + NODE_COUNT * NODE_FEATURES;
    const int GLOBAL_OFFSET = EDGE_OFFSET + NODE_COUNT * NODE_COUNT * EDGE_FEATURES;
    const int POLICY_HEAD = 384;
    const unsigned char* in = input + b * RECORD_BYTES;
    int* out = output + b * OUTPUT_INTS;
    for (int i = 0; i < OUTPUT_INTS; ++i) out[i] = 0;
    int node_count = eonego_clamp_i32((int)in[0], 0, NODE_COUNT);

    const float* node_w = weights + off[0]; const float* node_b = weights + off[1];
    const float* edge_w = weights + off[2]; const float* edge_b = weights + off[3];
    const float* glob_w = weights + off[4]; const float* glob_b = weights + off[5];
    const int HEAD = 24;
    const float* v0_w = weights + off[HEAD + 0]; const float* v0_b = weights + off[HEAD + 1];
    const float* v2_w = weights + off[HEAD + 2]; const float* v2_b = weights + off[HEAD + 3];
    const float* w0_w = weights + off[HEAD + 4]; const float* w0_b = weights + off[HEAD + 5];
    const float* w2_w = weights + off[HEAD + 6]; const float* w2_b = weights + off[HEAD + 7];
    const float* fh_w = weights + off[HEAD + 8]; const float* fh_b = weights + off[HEAD + 9];
    const float* th_w = weights + off[HEAD + 10]; const float* th_b = weights + off[HEAD + 11];

    float x[32 * 128];
    float nx[32 * 128];
    float g[128];
    float feat128[128];
    for (int n = 0; n < NODE_COUNT; ++n) {
        const unsigned char* src = in + NODE_OFFSET + n * NODE_FEATURES;
        for (int k = 0; k < NODE_FEATURES; ++k) feat128[k] = ((float)src[k]) * (1.0f / 255.0f);
        for (int o = 0; o < 128; ++o) {
            float y = node_b[o];
            for (int k = 0; k < NODE_FEATURES; ++k) y += node_w[o * NODE_FEATURES + k] * feat128[k];
            x[n * 128 + o] = n < node_count ? y : 0.0f;
        }
    }
    for (int k = 0; k < 32; ++k) feat128[k] = ((float)in[GLOBAL_OFFSET + k]) * (1.0f / 255.0f);
    for (int o = 0; o < 128; ++o) {
        float y = glob_b[o];
        for (int k = 0; k < 32; ++k) y += glob_w[o * 32 + k] * feat128[k];
        g[o] = y;
    }

    float accum[128], edgev[128], msg_in[384], upd_in[256], upd_out[128];
    for (int layer = 0; layer < 3; ++layer) {
        int L = 6 + layer * 6;
        const float* msg_w = weights + off[L + 0]; const float* msg_b = weights + off[L + 1];
        const float* upd_w = weights + off[L + 2]; const float* upd_b = weights + off[L + 3];
        const float* norm_w = weights + off[L + 4]; const float* norm_b = weights + off[L + 5];
        for (int dst = 0; dst < node_count; ++dst) {
            for (int k = 0; k < 128; ++k) accum[k] = 0.0f;
            const float* x_dst = x + dst * 128;
            for (int src_i = 0; src_i < node_count; ++src_i) {
                const float* x_src = x + src_i * 128;
                const unsigned char* er = in + EDGE_OFFSET + ((src_i * NODE_COUNT) + dst) * EDGE_FEATURES;
                for (int k = 0; k < EDGE_FEATURES; ++k) feat128[k] = ((float)er[k]) * (1.0f / 255.0f);
                for (int o = 0; o < 128; ++o) {
                    float y = edge_b[o];
                    for (int k = 0; k < EDGE_FEATURES; ++k) y += edge_w[o * EDGE_FEATURES + k] * feat128[k];
                    edgev[o] = y;
                }
                for (int k = 0; k < 128; ++k) {
                    msg_in[k] = x_src[k];
                    msg_in[128 + k] = x_dst[k];
                    msg_in[256 + k] = edgev[k];
                }
                for (int o = 0; o < 128; ++o) {
                    float y = eonego_linear_at(msg_w, msg_b, msg_in, o, 384);
                    if (y > 0.0f) accum[o] += y;
                }
            }
            for (int k = 0; k < 128; ++k) { upd_in[k] = x_dst[k]; upd_in[128 + k] = accum[k]; }
            for (int o = 0; o < 128; ++o) {
                float y = eonego_linear_at(upd_w, upd_b, upd_in, o, 256);
                upd_out[o] = y > 0.0f ? y : 0.0f;
                nx[dst * 128 + o] = x_dst[o] + upd_out[o];
            }
            eonego_ln128(nx + dst * 128, norm_w, norm_b);
        }
        for (int n = node_count; n < NODE_COUNT; ++n) for (int k = 0; k < 128; ++k) nx[n * 128 + k] = 0.0f;
        for (int i = 0; i < NODE_COUNT * 128; ++i) x[i] = nx[i];
    }

    float pooled[128];
    for (int k = 0; k < 128; ++k) pooled[k] = 0.0f;
    float denom = node_count > 0 ? (float)node_count : 1.0f;
    for (int n = 0; n < node_count; ++n) for (int k = 0; k < 128; ++k) pooled[k] += x[n * 128 + k] / denom;

    float h[256], hidden[128];
    for (int k = 0; k < 128; ++k) { h[k] = pooled[k]; h[128 + k] = g[k]; }
    for (int o = 0; o < 128; ++o) { float y = eonego_linear_at(v0_w, v0_b, h, o, 256); hidden[o] = y > 0.0f ? y : 0.0f; }
    float value = eonego_linear_at(v2_w, v2_b, hidden, 0, 128);
    out[0] = eonego_clamp_i32((int)roundf(value * (float)value_scale), -2500, 2500);
    for (int o = 0; o < 128; ++o) { float y = eonego_linear_at(w0_w, w0_b, h, o, 256); hidden[o] = y > 0.0f ? y : 0.0f; }
    eonego_wdl_softmax(
        eonego_linear_at(w2_w, w2_b, hidden, 0, 128),
        eonego_linear_at(w2_w, w2_b, hidden, 1, 128),
        eonego_linear_at(w2_w, w2_b, hidden, 2, 128),
        out);
    for (int i = 0; i < POLICY_HEAD; ++i) {
        out[4 + i] = eonego_scaled_logit(eonego_linear_at(fh_w, fh_b, h, i, 256));
        out[4 + POLICY_HEAD + i] = eonego_scaled_logit(eonego_linear_at(th_w, th_b, h, i, 256));
    }
}
)CUDA";

static bool compile_cuda_kernel(CudaBackend& b) {
    nvrtcProgram prog = nullptr;
    if (b.api.nvrtcCreateProgram(&prog, cuda_kernel_src, "eonego_graph_infer.cu", 0, nullptr, nullptr) != 0) {
        b.why = "nvrtcCreateProgram failed";
        return false;
    }

    const char* opts[] = { "--gpu-architecture=compute_52", "--std=c++11" };
    nvrtcResult cr = b.api.nvrtcCompileProgram(prog, 2, opts);
    if (cr != 0) {
        size_t log_size = 0;
        b.api.nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size + 1, 0);
        if (log_size > 0) b.api.nvrtcGetProgramLog(prog, log.data());
        b.why = std::string("NVRTC compile failed: ") + log.data();
        b.api.nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptx_size = 0;
    if (b.api.nvrtcGetPTXSize(prog, &ptx_size) != 0 || ptx_size == 0) {
        b.why = "nvrtcGetPTXSize failed";
        b.api.nvrtcDestroyProgram(&prog);
        return false;
    }

    std::vector<char> ptx(ptx_size);
    if (b.api.nvrtcGetPTX(prog, ptx.data()) != 0) {
        b.why = "nvrtcGetPTX failed";
        b.api.nvrtcDestroyProgram(&prog);
        return false;
    }
    b.api.nvrtcDestroyProgram(&prog);

    if (b.api.cuModuleLoadData(&b.module, ptx.data()) != 0) {
        b.why = "cuModuleLoadData failed";
        return false;
    }
    if (b.api.cuModuleGetFunction(&b.kernel, b.module, "eonego_graph_infer") != 0) {
        b.why = "cuModuleGetFunction failed";
        return false;
    }
    if (b.api.cuModuleGetFunction(&b.learned_kernel, b.module, "eonego_graphnet32_l1_infer") != 0) {
        b.why = "cuModuleGetFunction learned failed";
        return false;
    }
    if (b.api.cuModuleGetFunction(&b.learned_kernel128, b.module, "eonego_graphnet128_l3_infer") != 0) {
        b.why = "cuModuleGetFunction learned128 failed";
        return false;
    }
    return true;
}

static CudaBackend* create_cuda_backend(int device_id) {
    auto* b = new (std::nothrow) CudaBackend();
    if (b == nullptr) return nullptr;

    if (!load_cuda_api(b->api, b->why)) return b;
    if (b->api.cuInit(0) != 0) {
        b->why = "cuInit failed";
        return b;
    }
    CUdevice dev = 0;
    if (b->api.cuDeviceGet(&dev, device_id) != 0) {
        b->why = "cuDeviceGet failed";
        return b;
    }
    if (b->api.cuCtxCreate(&b->ctx, 0, dev) != 0) {
        b->why = "cuCtxCreate failed";
        return b;
    }
    if (!compile_cuda_kernel(*b)) return b;

    b->ready = true;
    return b;
}

static void destroy_cuda_backend(CudaBackend* b) {
    if (b == nullptr) return;
    if (b->d_output != 0 && b->api.cuMemFree != nullptr) b->api.cuMemFree(b->d_output);
    if (b->d_input != 0 && b->api.cuMemFree != nullptr) b->api.cuMemFree(b->d_input);
    if (b->d_offsets != 0 && b->api.cuMemFree != nullptr) b->api.cuMemFree(b->d_offsets);
    if (b->d_weights != 0 && b->api.cuMemFree != nullptr) b->api.cuMemFree(b->d_weights);
    if (b->module != nullptr && b->api.cuModuleUnload != nullptr) b->api.cuModuleUnload(b->module);
    if (b->ctx != nullptr && b->api.cuCtxDestroy != nullptr) b->api.cuCtxDestroy(b->ctx);
    if (b->api.nvrtc != nullptr) FreeLibrary(b->api.nvrtc);
    if (b->api.driver != nullptr) FreeLibrary(b->api.driver);
    delete b;
}

static bool ensure_cuda_scratch(CudaBackend* b, size_t input_bytes, size_t output_bytes) {
    if (b == nullptr || !b->ready) return false;
    if (b->api.cuCtxSetCurrent(b->ctx) != 0) return false;

    if (input_bytes > b->input_capacity) {
        if (b->d_input != 0) {
            b->api.cuMemFree(b->d_input);
            b->d_input = 0;
            b->input_capacity = 0;
        }
        if (b->api.cuMemAlloc(&b->d_input, input_bytes) != 0) return false;
        b->input_capacity = input_bytes;
    }

    if (output_bytes > b->output_capacity) {
        if (b->d_output != 0) {
            b->api.cuMemFree(b->d_output);
            b->d_output = 0;
            b->output_capacity = 0;
        }
        if (b->api.cuMemAlloc(&b->d_output, output_bytes) != 0) return false;
        b->output_capacity = output_bytes;
    }

    return true;
}

static bool infer_cuda(CudaBackend* b, const void* input_batch, void* output_batch, int batch_count) {
    if (b == nullptr || !b->ready || batch_count <= 0) return false;
    const size_t input_bytes = static_cast<size_t>(batch_count) * EOGPU_GRAPH_INPUT_BYTES;
    const size_t output_bytes = static_cast<size_t>(batch_count) * EOGPU_GRAPH_OUTPUT_BYTES;
    std::lock_guard<std::mutex> lock(b->mutex);
    if (!ensure_cuda_scratch(b, input_bytes, output_bytes)) return false;

    void* args[] = { &b->d_input, &b->d_output, &batch_count };
    const unsigned int block = 128;
    const unsigned int grid = static_cast<unsigned int>((batch_count + block - 1) / block);

    if (b->api.cuMemcpyHtoD(b->d_input, input_batch, input_bytes) != 0) return false;
    if (b->api.cuLaunchKernel(b->kernel, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr) != 0) return false;
    if (b->api.cuCtxSynchronize() != 0) return false;
    if (b->api.cuMemcpyDtoH(output_batch, b->d_output, output_bytes) != 0) return false;
    return true;
}

static bool infer_graphnet_cuda(CudaBackend* b, const void* input_batch, void* output_batch, int batch_count, int value_scale) {
    if (b == nullptr || !b->ready || !b->learned_ready || batch_count <= 0) return false;
    const size_t input_bytes = static_cast<size_t>(batch_count) * EOGPU_GRAPH_INPUT_BYTES;
    const size_t output_bytes = static_cast<size_t>(batch_count) * EOGPU_GRAPH_OUTPUT_BYTES;
    std::lock_guard<std::mutex> lock(b->mutex);
    if (!ensure_cuda_scratch(b, input_bytes, output_bytes)) return false;

    void* args[] = { &b->d_input, &b->d_weights, &b->d_offsets, &b->d_output, &batch_count, &value_scale };
    const unsigned int block = 128;
    const unsigned int grid = static_cast<unsigned int>((batch_count + block - 1) / block);

    if (b->api.cuMemcpyHtoD(b->d_input, input_batch, input_bytes) != 0) return false;
    if (b->api.cuLaunchKernel(b->learned_kernel, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr) != 0) return false;
    if (b->api.cuCtxSynchronize() != 0) return false;
    if (b->api.cuMemcpyDtoH(output_batch, b->d_output, output_bytes) != 0) return false;
    return true;
}
#else
struct CudaBackend {
    bool ready = false;
    bool learned_ready = false;
    std::string why = "CUDA dynamic backend is Windows-only in this build";
};

static CudaBackend* create_cuda_backend(int) { return new (std::nothrow) CudaBackend(); }
static void destroy_cuda_backend(CudaBackend* b) { delete b; }
static bool infer_cuda(CudaBackend*, const void*, void*, int) { return false; }
static bool infer_graphnet_cuda(CudaBackend*, const void*, void*, int, int) { return false; }
#endif

static bool read_exact(std::ifstream& f, void* dst, size_t n) {
    f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    return static_cast<size_t>(f.gcount()) == n;
}

static bool read_u32(std::ifstream& f, uint32_t& out) {
    unsigned char b[4];
    if (!read_exact(f, b, 4)) return false;
    out = static_cast<uint32_t>(b[0])
        | (static_cast<uint32_t>(b[1]) << 8)
        | (static_cast<uint32_t>(b[2]) << 16)
        | (static_cast<uint32_t>(b[3]) << 24);
    return true;
}

static bool read_u64(std::ifstream& f, uint64_t& out) {
    unsigned char b[8];
    if (!read_exact(f, b, 8)) return false;
    out = 0;
    for (int i = 7; i >= 0; --i) {
        out = (out << 8) | static_cast<uint64_t>(b[i]);
    }
    return true;
}

static bool skip_bytes(std::ifstream& f, uint64_t n) {
    if (n > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
    f.seekg(static_cast<std::streamoff>(n), std::ios::cur);
    return bool(f);
}

static const Tensor* find_tensor(const EonegoGpuHandle& h, const std::string& name) {
    auto it = h.tensor_index.find(name);
    if (it == h.tensor_index.end()) return nullptr;
    return &h.tensors[it->second];
}

static bool tensor_shape_eq(const Tensor* t, std::initializer_list<uint32_t> dims) {
    if (t == nullptr || t->dims.size() != dims.size()) return false;
    size_t i = 0;
    for (uint32_t d : dims) {
        if (t->dims[i++] != d) return false;
    }
    return true;
}

static bool require_tensor(const EonegoGpuHandle& h, const std::string& name, std::initializer_list<uint32_t> dims) {
    return tensor_shape_eq(find_tensor(h, name), dims);
}

static bool validate_graphnet_tensors(const EonegoGpuHandle& h, std::string& why) {
    const uint32_t d = h.d_model;

    auto req = [&](const std::string& name, std::initializer_list<uint32_t> dims) -> bool {
        if (!require_tensor(h, name, dims)) {
            std::ostringstream oss;
            oss << "missing or bad tensor shape: " << name;
            why = oss.str();
            return false;
        }
        return true;
    };

    if (!req("node_in.weight", { d, NODE_FEATURES }) || !req("node_in.bias", { d })
        || !req("edge_in.weight", { d, EDGE_FEATURES }) || !req("edge_in.bias", { d })
        || !req("global_in.weight", { d, 32 }) || !req("global_in.bias", { d })) {
        return false;
    }

    for (uint32_t i = 0; i < h.layers; ++i) {
        const std::string p = std::to_string(i);
        if (!req("msg." + p + ".weight", { d, d * 3 }) || !req("msg." + p + ".bias", { d })
            || !req("upd." + p + ".weight", { d, d * 2 }) || !req("upd." + p + ".bias", { d })
            || !req("norm." + p + ".weight", { d }) || !req("norm." + p + ".bias", { d })) {
            return false;
        }
    }

    return req("value.0.weight", { d, d * 2 }) && req("value.0.bias", { d })
        && req("value.2.weight", { 1, d }) && req("value.2.bias", { 1 })
        && req("wdl.0.weight", { d, d * 2 }) && req("wdl.0.bias", { d })
        && req("wdl.2.weight", { 3, d }) && req("wdl.2.bias", { 3 })
        && req("from_head.weight", { POLICY_HEAD, d * 2 }) && req("from_head.bias", { POLICY_HEAD })
        && req("to_head.weight", { POLICY_HEAD, d * 2 }) && req("to_head.bias", { POLICY_HEAD });
}

static bool graphnet_cuda_shape_supported(const EonegoGpuHandle& h) {
    return h.has_graphnet_weights && ((h.d_model == 32 && h.layers == 1) || (h.d_model == 128 && h.layers == 3));
}

#ifdef _WIN32
static bool upload_graphnet_cuda_weights(EonegoGpuHandle& h) {
    if (h.cuda == nullptr || !h.cuda->ready || !graphnet_cuda_shape_supported(h)) return false;
    CudaBackend* b = h.cuda;
    if (b->api.cuCtxSetCurrent(b->ctx) != 0) return false;

    std::vector<std::string> names = {
        "node_in.weight", "node_in.bias", "edge_in.weight", "edge_in.bias",
        "global_in.weight", "global_in.bias"
    };
    for (uint32_t layer = 0; layer < h.layers; ++layer) {
        const std::string p = std::to_string(layer);
        names.push_back("msg." + p + ".weight");
        names.push_back("msg." + p + ".bias");
        names.push_back("upd." + p + ".weight");
        names.push_back("upd." + p + ".bias");
        names.push_back("norm." + p + ".weight");
        names.push_back("norm." + p + ".bias");
    }
    names.push_back("value.0.weight");
    names.push_back("value.0.bias");
    names.push_back("value.2.weight");
    names.push_back("value.2.bias");
    names.push_back("wdl.0.weight");
    names.push_back("wdl.0.bias");
    names.push_back("wdl.2.weight");
    names.push_back("wdl.2.bias");
    names.push_back("from_head.weight");
    names.push_back("from_head.bias");
    names.push_back("to_head.weight");
    names.push_back("to_head.bias");

    std::vector<int> offsets(names.size(), 0);
    std::vector<float> flat;

    for (size_t i = 0; i < names.size(); ++i) {
        const Tensor* t = find_tensor(h, names[i]);
        if (t == nullptr) return false;
        offsets[i] = static_cast<int>(flat.size());
        flat.insert(flat.end(), t->data.begin(), t->data.end());
    }

    b->weight_bytes = flat.size() * sizeof(float);
    b->offset_bytes = offsets.size() * sizeof(int);
    if (b->api.cuMemAlloc(&b->d_weights, b->weight_bytes) != 0) return false;
    if (b->api.cuMemAlloc(&b->d_offsets, b->offset_bytes) != 0) return false;
    if (b->api.cuMemcpyHtoD(b->d_weights, flat.data(), b->weight_bytes) != 0) return false;
    if (b->api.cuMemcpyHtoD(b->d_offsets, offsets.data(), b->offset_bytes) != 0) return false;
    b->learned_kernel = (h.d_model == 128 && h.layers == 3) ? b->learned_kernel128 : b->learned_kernel;
    b->learned_ready = true;
    return true;
}
#else
static bool upload_graphnet_cuda_weights(EonegoGpuHandle&) { return false; }
#endif

static bool parse_model(const char* path, EonegoGpuHandle& out, char* err_buf, int err_len) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        set_error(err_buf, err_len, "could not open EONGR01 model");
        return false;
    }

    char magic[8];
    if (!read_exact(f, magic, 8) || std::memcmp(magic, "EONGR01!", 8) != 0) {
        set_error(err_buf, err_len, "bad EONGR01 magic");
        return false;
    }

    uint32_t version, schema, d_model, layers, flags, value_scale, reserved;
    if (!read_u32(f, version) || !read_u32(f, schema) || !read_u32(f, d_model) || !read_u32(f, layers)
        || !read_u32(f, flags) || !read_u32(f, value_scale) || !read_u32(f, reserved)) {
        set_error(err_buf, err_len, "truncated EONGR01 header");
        return false;
    }

    if (version != EONGR_VERSION || schema != EONGR_SCHEMA || d_model < 32 || d_model > 1024
        || layers < 1 || layers > 16 || value_scale == 0 || reserved != 0) {
        set_error(err_buf, err_len, "unsupported EONGR01 header");
        return false;
    }

    out.d_model = d_model;
    out.layers = layers;
    out.value_scale = value_scale;

    char weights_magic[8];
    if (!read_exact(f, weights_magic, 8) || std::memcmp(weights_magic, "EONGRW1\0", 8) != 0) {
        set_error(err_buf, err_len, "missing EONGR01 tensor table");
        return false;
    }

    uint32_t tensor_count;
    if (!read_u32(f, tensor_count) || tensor_count == 0 || tensor_count > 4096) {
        set_error(err_buf, err_len, "invalid EONGR01 tensor count");
        return false;
    }

    for (uint32_t t = 0; t < tensor_count; ++t) {
        uint32_t name_len, dtype, rank;
        if (!read_u32(f, name_len) || name_len == 0 || name_len > 512) {
            set_error(err_buf, err_len, "invalid tensor name");
            return false;
        }
        std::string name(name_len, '\0');
        if (!read_exact(f, name.data(), name_len) || !read_u32(f, dtype) || !read_u32(f, rank) || dtype != DTYPE_F32 || rank > 4) {
            set_error(err_buf, err_len, "invalid tensor header");
            return false;
        }

        uint64_t elems = 1;
        std::vector<uint32_t> dims;
        dims.reserve(rank);
        for (uint32_t i = 0; i < rank; ++i) {
            uint32_t dim;
            if (!read_u32(f, dim) || dim == 0 || dim > 1'000'000) {
                set_error(err_buf, err_len, "invalid tensor shape");
                return false;
            }
            dims.push_back(dim);
            elems *= dim;
            if (elems > (1ull << 34)) {
                set_error(err_buf, err_len, "tensor too large");
                return false;
            }
        }

        uint64_t nbytes;
        if (!read_u64(f, nbytes) || nbytes != elems * sizeof(float)) {
            set_error(err_buf, err_len, "invalid tensor data");
            return false;
        }
        if (out.tensor_index.find(name) != out.tensor_index.end()) {
            set_error(err_buf, err_len, "duplicate tensor name");
            return false;
        }

        Tensor tensor;
        tensor.name = std::move(name);
        tensor.dims = std::move(dims);
        tensor.data.resize(static_cast<size_t>(elems));
        if (!read_exact(f, tensor.data.data(), static_cast<size_t>(nbytes))) {
            set_error(err_buf, err_len, "truncated tensor data");
            return false;
        }

        out.tensor_index.emplace(tensor.name, out.tensors.size());
        out.tensors.push_back(std::move(tensor));
    }

    out.tensor_count = tensor_count;
    std::string why;
    out.has_graphnet_weights = validate_graphnet_tensors(out, why);
    (void)flags;
    return true;
}

static int piece_value(int pt) {
    switch (pt) {
    case 0: return 100;
    case 1: return 320;
    case 2: return 330;
    case 3: return 500;
    case 4: return 900;
    default: return 0;
    }
}

static float linear_at(const Tensor& w, const Tensor& b, const float* x, uint32_t out_idx, uint32_t in_dim) {
    const float* row = w.data.data() + static_cast<size_t>(out_idx) * in_dim;
    float y = b.data[out_idx];
    for (uint32_t i = 0; i < in_dim; ++i) y += row[i] * x[i];
    return y;
}

static void linear_vec(const Tensor& w, const Tensor& b, const float* x, uint32_t out_dim, uint32_t in_dim, float* y) {
    for (uint32_t o = 0; o < out_dim; ++o) y[o] = linear_at(w, b, x, o, in_dim);
}

static void relu_vec(float* x, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

static void layer_norm_inplace(float* x, uint32_t d, const Tensor& weight, const Tensor& bias) {
    float mean = 0.0f;
    for (uint32_t i = 0; i < d; ++i) mean += x[i];
    mean /= static_cast<float>(d);

    float var = 0.0f;
    for (uint32_t i = 0; i < d; ++i) {
        const float z = x[i] - mean;
        var += z * z;
    }
    var /= static_cast<float>(d);

    const float inv = 1.0f / std::sqrt(var + 1.0e-5f);
    for (uint32_t i = 0; i < d; ++i) x[i] = (x[i] - mean) * inv * weight.data[i] + bias.data[i];
}

static int scaled_logit(float x) {
    const float y = std::round(x * 1024.0f);
    if (y > 2147483647.0f) return 2147483647;
    if (y < -2147483648.0f) return static_cast<int>(-2147483647 - 1);
    return static_cast<int>(y);
}

static int scaled_value(float x, uint32_t value_scale) {
    const float y = std::round(x * static_cast<float>(value_scale));
    if (y > 2500.0f) return 2500;
    if (y < -2500.0f) return -2500;
    return static_cast<int>(y);
}

static void write_wdl_softmax(const float* logits, int32_t* out) {
    const float m = std::max(logits[0], std::max(logits[1], logits[2]));
    const float ew = std::exp(logits[0] - m);
    const float ed = std::exp(logits[1] - m);
    const float el = std::exp(logits[2] - m);
    const float s = ew + ed + el;
    int w = static_cast<int>(std::round(1000.0f * ew / s));
    int d = static_cast<int>(std::round(1000.0f * ed / s));
    int l = 1000 - w - d;
    if (l < 0) {
        l = 0;
        if (w >= d) --w; else --d;
    }
    out[1] = clamp_i32(w, 0, 1000);
    out[2] = clamp_i32(d, 0, 1000 - out[1]);
    out[3] = 1000 - out[1] - out[2];
}

static bool infer_graphnet_one(const EonegoGpuHandle& h, const uint8_t* in, int32_t* out) {
    if (!h.has_graphnet_weights) return false;

    std::memset(out, 0, EOGPU_GRAPH_OUTPUT_BYTES);
    const uint32_t d = h.d_model;
    const int node_count_i = clamp_i32(static_cast<int>(in[0]), 0, NODE_COUNT);
    const uint32_t node_count = static_cast<uint32_t>(node_count_i);
    const float inv255 = 1.0f / 255.0f;

    auto tensor = [&](const std::string& name) -> const Tensor& { return *find_tensor(h, name); };

    std::vector<float> x(static_cast<size_t>(NODE_COUNT) * d, 0.0f);
    std::vector<float> edge(static_cast<size_t>(NODE_COUNT) * NODE_COUNT * d, 0.0f);
    std::vector<float> g(d, 0.0f);
    std::vector<float> feat(std::max<uint32_t>(32, std::max<uint32_t>(NODE_FEATURES, EDGE_FEATURES)), 0.0f);

    const Tensor& node_w = tensor("node_in.weight");
    const Tensor& node_b = tensor("node_in.bias");
    for (uint32_t n = 0; n < NODE_COUNT; ++n) {
        const uint8_t* src = in + NODE_OFFSET + n * NODE_FEATURES;
        for (uint32_t i = 0; i < NODE_FEATURES; ++i) feat[i] = static_cast<float>(src[i]) * inv255;
        linear_vec(node_w, node_b, feat.data(), d, NODE_FEATURES, x.data() + static_cast<size_t>(n) * d);
        if (n >= node_count) std::fill(x.begin() + static_cast<size_t>(n) * d, x.begin() + static_cast<size_t>(n + 1) * d, 0.0f);
    }

    const Tensor& edge_w = tensor("edge_in.weight");
    const Tensor& edge_b = tensor("edge_in.bias");
    for (uint32_t i = 0; i < NODE_COUNT; ++i) {
        for (uint32_t j = 0; j < NODE_COUNT; ++j) {
            const uint8_t* src = in + EDGE_OFFSET + ((i * NODE_COUNT) + j) * EDGE_FEATURES;
            for (uint32_t k = 0; k < EDGE_FEATURES; ++k) feat[k] = static_cast<float>(src[k]) * inv255;
            linear_vec(edge_w, edge_b, feat.data(), d, EDGE_FEATURES, edge.data() + (static_cast<size_t>(i) * NODE_COUNT + j) * d);
        }
    }

    const Tensor& global_w = tensor("global_in.weight");
    const Tensor& global_b = tensor("global_in.bias");
    const uint8_t* globals = in + NODE_OFFSET + NODE_COUNT * NODE_FEATURES + NODE_COUNT * NODE_COUNT * EDGE_FEATURES;
    for (uint32_t i = 0; i < 32; ++i) feat[i] = static_cast<float>(globals[i]) * inv255;
    linear_vec(global_w, global_b, feat.data(), d, 32, g.data());

    std::vector<float> accum(d, 0.0f);
    std::vector<float> msg_in(static_cast<size_t>(d) * 3, 0.0f);
    std::vector<float> upd_in(static_cast<size_t>(d) * 2, 0.0f);
    std::vector<float> upd_out(d, 0.0f);
    std::vector<float> next_x(static_cast<size_t>(NODE_COUNT) * d, 0.0f);

    for (uint32_t layer = 0; layer < h.layers; ++layer) {
        const std::string p = std::to_string(layer);
        const Tensor& msg_w = tensor("msg." + p + ".weight");
        const Tensor& msg_b = tensor("msg." + p + ".bias");
        const Tensor& upd_w = tensor("upd." + p + ".weight");
        const Tensor& upd_b = tensor("upd." + p + ".bias");
        const Tensor& norm_w = tensor("norm." + p + ".weight");
        const Tensor& norm_b = tensor("norm." + p + ".bias");
        std::fill(next_x.begin(), next_x.end(), 0.0f);

        for (uint32_t dst = 0; dst < node_count; ++dst) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            const float* x_dst = x.data() + static_cast<size_t>(dst) * d;

            for (uint32_t src = 0; src < node_count; ++src) {
                const float* x_src = x.data() + static_cast<size_t>(src) * d;
                const float* e = edge.data() + (static_cast<size_t>(src) * NODE_COUNT + dst) * d;
                std::memcpy(msg_in.data(), x_src, sizeof(float) * d);
                std::memcpy(msg_in.data() + d, x_dst, sizeof(float) * d);
                std::memcpy(msg_in.data() + static_cast<size_t>(2) * d, e, sizeof(float) * d);

                for (uint32_t k = 0; k < d; ++k) {
                    float v = linear_at(msg_w, msg_b, msg_in.data(), k, d * 3);
                    if (v > 0.0f) accum[k] += v;
                }
            }

            std::memcpy(upd_in.data(), x_dst, sizeof(float) * d);
            std::memcpy(upd_in.data() + d, accum.data(), sizeof(float) * d);
            linear_vec(upd_w, upd_b, upd_in.data(), d, d * 2, upd_out.data());
            relu_vec(upd_out.data(), d);

            float* y = next_x.data() + static_cast<size_t>(dst) * d;
            for (uint32_t k = 0; k < d; ++k) y[k] = x_dst[k] + upd_out[k];
            layer_norm_inplace(y, d, norm_w, norm_b);
        }

        x.swap(next_x);
    }

    std::vector<float> pooled(d, 0.0f);
    const float denom = static_cast<float>(std::max<uint32_t>(node_count, 1));
    for (uint32_t n = 0; n < node_count; ++n) {
        const float* xn = x.data() + static_cast<size_t>(n) * d;
        for (uint32_t k = 0; k < d; ++k) pooled[k] += xn[k] / denom;
    }

    std::vector<float> hvec(static_cast<size_t>(d) * 2, 0.0f);
    std::memcpy(hvec.data(), pooled.data(), sizeof(float) * d);
    std::memcpy(hvec.data() + d, g.data(), sizeof(float) * d);

    std::vector<float> hidden(d, 0.0f);
    linear_vec(tensor("value.0.weight"), tensor("value.0.bias"), hvec.data(), d, d * 2, hidden.data());
    relu_vec(hidden.data(), d);
    const float value = linear_at(tensor("value.2.weight"), tensor("value.2.bias"), hidden.data(), 0, d);
    out[0] = scaled_value(value, h.value_scale);

    linear_vec(tensor("wdl.0.weight"), tensor("wdl.0.bias"), hvec.data(), d, d * 2, hidden.data());
    relu_vec(hidden.data(), d);
    float wdl_logits[3] = {
        linear_at(tensor("wdl.2.weight"), tensor("wdl.2.bias"), hidden.data(), 0, d),
        linear_at(tensor("wdl.2.weight"), tensor("wdl.2.bias"), hidden.data(), 1, d),
        linear_at(tensor("wdl.2.weight"), tensor("wdl.2.bias"), hidden.data(), 2, d)
    };
    write_wdl_softmax(wdl_logits, out);

    const Tensor& from_w = tensor("from_head.weight");
    const Tensor& from_b = tensor("from_head.bias");
    const Tensor& to_w = tensor("to_head.weight");
    const Tensor& to_b = tensor("to_head.bias");
    for (uint32_t i = 0; i < POLICY_HEAD; ++i) {
        out[4 + i] = scaled_logit(linear_at(from_w, from_b, hvec.data(), i, d * 2));
        out[4 + POLICY_HEAD + i] = scaled_logit(linear_at(to_w, to_b, hvec.data(), i, d * 2));
    }

    return true;
}

static void write_wdl(int value_cp, int32_t* out) {
    const int abs_cp = std::abs(value_cp);
    int draw = clamp_i32(360 - abs_cp / 2, 40, 620);
    int decisive = 1000 - draw;
    int win = decisive / 2 + clamp_i32(value_cp / 4, -decisive / 2, decisive / 2);
    win = clamp_i32(win, 0, 1000 - draw);
    int loss = 1000 - draw - win;
    out[1] = win;
    out[2] = draw;
    out[3] = loss;
}

static int rel_center_bonus(int rel_sq) {
    const int f = rel_sq & 7;
    const int r = rel_sq >> 3;
    const int df = std::abs(f - 3) < std::abs(f - 4) ? std::abs(f - 3) : std::abs(f - 4);
    const int dr = std::abs(r - 3) < std::abs(r - 4) ? std::abs(r - 3) : std::abs(r - 4);
    return 18 - 4 * (df + dr);
}

static void infer_one(const uint8_t* in, int32_t* out) {
    std::memset(out, 0, EOGPU_GRAPH_OUTPUT_BYTES);
    const int node_count = clamp_i32(static_cast<int>(in[0]), 0, NODE_COUNT);
    const bool in_check = in[2] != 0;
    const int checker_count = static_cast<int>(in[3]);
    int score = 0;
    int own_bishops = 0;
    int opp_bishops = 0;

    for (int i = 0; i < node_count; ++i) {
        const uint8_t* n = in + NODE_OFFSET + i * NODE_FEATURES;
        if (n[0] == 0) continue;

        const bool own = n[1] != 0;
        const int pt = static_cast<int>(n[3]);
        const int rel_sq = static_cast<int>(n[5]);
        const int rel_rank = rel_sq >> 3;
        int v = piece_value(pt);

        v += rel_center_bonus(rel_sq);
        if (pt == 0) v += rel_rank * 7;
        if (n[8] != 0) v -= 18;
        if (n[9] != 0) v += 10;
        if (n[10] != 0) v += 28;

        if (pt == 2) {
            if (own) ++own_bishops; else ++opp_bishops;
        }

        score += own ? v : -v;

        if (pt >= 0 && pt < 6) {
            const int idx = pt * 64 + rel_sq;
            if (idx >= 0 && idx < POLICY_HEAD) {
                out[4 + idx] += own ? 900 + v / 2 : -350;
            }
        }
    }

    if (own_bishops >= 2) score += 35;
    if (opp_bishops >= 2) score -= 35;
    if (in_check) score -= 55 + 30 * checker_count;

    for (int i = 0; i < node_count; ++i) {
        const uint8_t* a = in + NODE_OFFSET + i * NODE_FEATURES;
        if (a[0] == 0 || a[1] == 0) continue;

        const int apt = static_cast<int>(a[3]);
        const int arel = static_cast<int>(a[5]);
        if (apt < 0 || apt >= 6 || arel < 0 || arel >= 64) continue;

        for (int j = 0; j < node_count; ++j) {
            if (i == j) continue;
            const uint8_t* b = in + NODE_OFFSET + j * NODE_FEATURES;
            if (b[0] == 0) continue;

            const uint8_t* e = in + EDGE_OFFSET + ((i * NODE_COUNT) + j) * EDGE_FEATURES;
            if (e[0] == 0) continue;

            const bool attacks = e[2] != 0;
            const bool same_color = e[1] != 0;
            const int bpt = static_cast<int>(b[3]);
            const int brel = static_cast<int>(b[5]);

            if (attacks && !same_color) {
                score += clamp_i32(piece_value(bpt) / 16, 4, 60);
                const int to_idx = apt * 64 + brel;
                if (to_idx >= 0 && to_idx < POLICY_HEAD) {
                    out[4 + POLICY_HEAD + to_idx] += 600 + piece_value(bpt) / 2;
                }
            } else if (attacks && same_color) {
                const int to_idx = apt * 64 + brel;
                if (to_idx >= 0 && to_idx < POLICY_HEAD) {
                    out[4 + POLICY_HEAD + to_idx] += 35;
                }
            }
        }
    }

    out[0] = clamp_i32(score, -2500, 2500);
    write_wdl(out[0], out);
}

extern "C" EOGPU_API void* eogpu_create(const char* model_path, int device_id, int max_batch, char* err_buf, int err_len) {
    if (model_path == nullptr || max_batch <= 0) {
        set_error(err_buf, err_len, "invalid eogpu_create arguments");
        return nullptr;
    }
    try {
        auto* h = new EonegoGpuHandle();
        h->model_path = model_path;
        h->device_id = device_id;
        h->max_batch = max_batch;
        h->cuda = nullptr;
        if (!parse_model(model_path, *h, err_buf, err_len)) {
            delete h;
            return nullptr;
        }

        const char* disable_cuda = std::getenv("EONEGO_GRAPH_CUDA");
        if (disable_cuda == nullptr || std::strcmp(disable_cuda, "0") != 0) {
            h->cuda = create_cuda_backend(device_id);
            upload_graphnet_cuda_weights(*h);
        }

        h->infer_ready = true;
        set_error(err_buf, err_len, "");
        return h;
    } catch (const std::bad_alloc&) {
        set_error(err_buf, err_len, "out of memory");
        return nullptr;
    }
}

extern "C" EOGPU_API int eogpu_infer(void* handle, const void* input_batch, void* output_batch, int batch_count) {
    auto* h = static_cast<EonegoGpuHandle*>(handle);
    if (h == nullptr || !h->infer_ready) return -2;
    if (batch_count == 0) return 0;
    if (batch_count < 0 || batch_count > h->max_batch || input_batch == nullptr || output_batch == nullptr) return -1;

    const auto* in = static_cast<const uint8_t*>(input_batch);
    auto* out = static_cast<int32_t*>(output_batch);
    const char* disable_learned = std::getenv("EONEGO_GRAPH_LEARNED");
    const bool use_learned = h->has_graphnet_weights && (disable_learned == nullptr || std::strcmp(disable_learned, "0") != 0);

    if (use_learned) {
        if (h->cuda != nullptr && h->cuda->learned_ready && infer_graphnet_cuda(h->cuda, input_batch, output_batch, batch_count, static_cast<int>(h->value_scale))) {
            return 0;
        }
        for (int b = 0; b < batch_count; ++b) {
            if (!infer_graphnet_one(*h, in + b * RECORD_BYTES, out + b * OUTPUT_INTS)) return -1;
        }
        return 0;
    }

    if (h->cuda != nullptr && h->cuda->ready && infer_cuda(h->cuda, input_batch, output_batch, batch_count)) {
        return 0;
    }

    for (int b = 0; b < batch_count; ++b) {
        infer_one(in + b * RECORD_BYTES, out + b * OUTPUT_INTS);
    }

    return 0;
}

extern "C" EOGPU_API int eogpu_backend(void* handle) {
    auto* h = static_cast<EonegoGpuHandle*>(handle);
    if (h == nullptr) return -1;
    return (h->cuda != nullptr && h->cuda->ready) ? 1 : 0;
}

extern "C" EOGPU_API int eogpu_model_status(void* handle) {
    auto* h = static_cast<EonegoGpuHandle*>(handle);
    if (h == nullptr) return -1;
    if (!h->has_graphnet_weights) return 0;
    return (h->cuda != nullptr && h->cuda->learned_ready) ? 2 : 1;
}

extern "C" EOGPU_API void eogpu_destroy(void* handle) {
    auto* h = static_cast<EonegoGpuHandle*>(handle);
    if (h != nullptr) {
        destroy_cuda_backend(h->cuda);
        delete h;
    }
}
