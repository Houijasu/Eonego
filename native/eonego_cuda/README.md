# eonego_cuda

Native C ABI boundary for Eonego's graph evaluator.

Current state:

- `eonego_cuda.h` is the stable C ABI intended for NativeAOT-friendly P/Invoke from `Eonego.GraphGpu`.
- `eogpu_create` validates the EONGR01 header and explicit EONGRW1 tensor table.
- `eogpu_infer` accepts native batches (`batch_count > 1`) and runs CUDA first when `nvcuda.dll` and NVRTC
  are available. The CUDA path compiles its kernel dynamically at `eogpu_create` time, so this MinGW build
  does not require `nvcc` or MSVC `cl.exe`.
- The EONGR01 tensor table is parsed into native memory. The runtime validates the complete GraphNet
  state-dict contract exported by `trainer/graph_export.py` (`node_in`, `edge_in`, `global_in`, every
  `msg/upd/norm` layer, value/WDL heads, and 384-wide from/to policy heads).
- When that complete GraphNet contract is present, inference uses a native learned GraphNet forward pass:
  node/edge/global linear projections, relation-message layers, layer norm, value/WDL heads, and 384-wide
  from/to policy heads. Value is scaled by the EONGR01 `valueScale`; WDL is softmaxed to permille; policy
  logits are scaled to engine integers.
- Compact `d_model=32, layers=1` and default `d_model=128, layers=3` GraphNet exports use learned CUDA
  kernels when CUDA is available. Other complete GraphNet shapes use the learned CPU-native path until
  generic CUDA kernels are implemented.
- If CUDA initialization or kernel launch fails, the same DLL falls back to a deterministic scalar CPU
  graph-feature backend. That keeps CPU search, model-container validation, batched P/Invoke, fallback
  behavior, and policy plumbing testable on machines without a CUDA runtime.
- CUDA input/output scratch buffers are persistent per native handle and grow to the largest observed batch,
  so repeated search batches do not allocate/free device buffers on every inference call. CUDA calls are
  serialized per handle because the context, scratch buffers, and uploaded weight table are shared by CPU
  search threads.
- The CUDA path still runs the deterministic bring-up graph-feature kernel when no complete GraphNet
  weights are present, or when `EONEGO_GRAPH_LEARNED=0` disables learned execution.
- `eogpu_backend` reports the selected backend: `1` CUDA, `0` CPU scalar fallback.
- `eogpu_model_status` reports whether the loaded tensor table is a complete GraphNet contract.

Build smoke:

```powershell
cmake -S native/eonego_cuda -B artifacts/eonego_cuda_build
cmake --build artifacts/eonego_cuda_build --config Release
```

Put `artifacts/eonego_cuda_build` on `PATH`, or copy `eonego_cuda.dll` next to the engine executable, before
running with `EONEGO_GRAPH=<model.eongr01>`.

Runtime gates in the engine:

- `EONEGO_GRAPH=<path>` loads an EONGR01 graph model header.
- `EONEGO_GRAPH_MODE=leaf|policy|all`, default `leaf`. `policy`/`all` feed graph logits into the existing
  negamax move-ordering and LMR policy slots; no MCTS is involved.
- `EONEGO_GRAPH_DEVICE`, `EONEGO_GRAPH_BATCH`, and `EONEGO_GRAPH_WAIT_US` configure the native batcher.
- `EONEGO_GRAPH_CUDA=0` forces the scalar CPU backend; any other value leaves CUDA enabled.
- `EONEGO_GRAPH_LEARNED=0` disables the learned GraphNet tensor forward path and uses the bring-up graph
  feature evaluator instead.

Next step: generalize learned CUDA beyond the currently specialized `d_model=32/layers=1` and
`d_model=128/layers=3` runtime shapes, then optimize the serial-per-position kernels into parallel graph
layers.
