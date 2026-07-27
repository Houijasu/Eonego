#pragma once

#ifdef _WIN32
#define EOGPU_API __declspec(dllexport)
#else
#define EOGPU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define EOGPU_GRAPH_INPUT_BYTES 8612
#define EOGPU_GRAPH_OUTPUT_BYTES 3088

EOGPU_API void* eogpu_create(const char* model_path, int device_id, int max_batch, char* err_buf, int err_len);
// Returns 0 on success. A zero batch is a readiness probe; return 0 only when inference is available.
// Return -2 only when the loaded library cannot perform inference, so the engine keeps NNUE fallback enabled.
EOGPU_API int eogpu_infer(void* handle, const void* input_batch, void* output_batch, int batch_count);
// Returns 1 for CUDA, 0 for CPU scalar fallback, and -1 for an invalid handle.
EOGPU_API int eogpu_backend(void* handle);
// Returns 2 when complete GraphNet weights are ready for learned CUDA, 1 when complete weights are
// available for learned CPU, 0 for partial/unknown, and -1 for an invalid handle.
EOGPU_API int eogpu_model_status(void* handle);
EOGPU_API void eogpu_destroy(void* handle);

#ifdef __cplusplus
}
#endif
