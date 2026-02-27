# Plan: SGLang-Style Torch Distribution Backend for vLLM (Ray Alternative) + TRT-LLM Interop

## Status

Proposed design and implementation plan.

## Objective

Add a **new vLLM distributed backend** that behaves like SGLang’s process model (Torch/NCCL + IPC control plane), as an alternative to Ray.

Primary goals:

1. Keep vLLM model/runtime strengths.
2. Remove Ray as a hard dependency for multi-process/multi-node orchestration.
3. Provide a clean interoperability model with TRT-LLM deployments.

## Why this approach

SGLang’s runtime architecture is explicitly process-oriented and IPC-first:

- HTTP + tokenizer in main process; scheduler/detokenizer as subprocesses in [`launch_server()`](sglang/python/sglang/srt/entrypoints/http_server.py:1819).
- Engine bootstraps subprocesses and ZMQ links in [`Engine.__init__()`](sglang/python/sglang/srt/entrypoints/engine.py:139).
- DP fanout/control happens in a dedicated controller with load-balancing in [`DataParallelController`](sglang/python/sglang/srt/managers/data_parallel_controller.py:116) and dispatch loop in [`event_loop()`](sglang/python/sglang/srt/managers/data_parallel_controller.py:562).

vLLM already has extensibility points for this:

- backend selection in [`Executor.get_class()`](tmp/vllm/vllm/v1/executor/abstract.py:47),
- process-based backend in [`MultiprocExecutor`](tmp/vllm/vllm/v1/executor/multiproc_executor.py:94),
- external launcher path in [`ExecutorWithExternalLauncher`](tmp/vllm/vllm/v1/executor/uniproc_executor.py:140),
- backend configuration/validation in [`ParallelConfig`](tmp/vllm/vllm/config/parallel.py:605).

## Target Architecture (New Backend)

Introduce `distributed_executor_backend="sgipc"` (name can be adjusted), implemented as a new executor class.

### High-level components

1. **Engine process (existing vLLM scheduler/engine)**
   - keeps scheduling logic unchanged.
2. **Control-plane dispatcher (new, optional process)**
   - SGLang-like request routing/load balancing across DP groups.
3. **Worker processes (existing vLLM worker wrappers)**
   - execute model and communicate using Torch distributed + local IPC.
4. **IPC fabric**
   - ZMQ (or existing message queues where possible) for control messages and health/load updates.
5. **Data plane**
   - torch.distributed/NCCL for collective ops only.

## Key Design Principles

1. **No scheduler semantic drift**: preserve vLLM scheduling behavior.
2. **Control/data plane split**: IPC for orchestration, NCCL for tensors.
3. **Deterministic failure handling**: explicit worker liveness + fail-fast callbacks.
4. **Incremental rollout**: single-node first, then multi-node.

## Phase Plan

### Phase 0 — RFC + invariants

Define hard invariants before coding:

- backend must pass all `mp` executor functional tests,
- no regression for `ray`, `mp`, `uni`, `external_launcher`,
- same outputs for deterministic seeds across `mp` and `sgipc`.

Deliverable: design doc + interface contract.

---

### Phase 1 — Backend plumbing in vLLM

1. Add backend keyword in config validation (near [`ParallelConfig._verify_args()`](tmp/vllm/vllm/config/parallel.py:674)).
2. Register backend selection in [`Executor.get_class()`](tmp/vllm/vllm/v1/executor/abstract.py:47).
3. Scaffold new class, e.g. `SgIpcExecutor`, reusing safe pieces from [`MultiprocExecutor`](tmp/vllm/vllm/v1/executor/multiproc_executor.py:94).

Deliverable: backend can boot/teardown and pass smoke health checks.

---

### Phase 2 — Single-node control plane (SGLang-style)

1. Add local dispatcher loop inspired by SGLang’s [`DataParallelController.event_loop()`](sglang/python/sglang/srt/managers/data_parallel_controller.py:562).
2. Implement load-balancing policies analogous to round-robin and load-aware dispatch (request count/token count).
3. Feed scheduler load metrics into dispatcher.
4. Keep existing vLLM RPC/worker method interfaces untouched.

Deliverable: single-node TP/PP/DP works without Ray.

---

### Phase 3 — Multi-node `sgipc`

1. Add bootstrap/port handshake similar to SGLang node-rank coordination in [`_broadcast_worker_ports()`](sglang/python/sglang/srt/managers/data_parallel_controller.py:284).
2. Replace loopback-only init assumptions in `mp` path with explicit master addr/port per DP group where needed.
3. Add node-local worker leaders and cross-node control relay.
4. Harden startup timeouts and retry behavior.

Deliverable: multi-node execution without Ray with explicit torch+IPC launch procedure.

---

### Phase 4 — Production hardening

1. Worker monitor parity with [`start_worker_monitor()`](tmp/vllm/vllm/v1/executor/multiproc_executor.py:245).
2. Structured health endpoints and per-rank status.
3. Graceful drain + restart semantics.
4. Perf tuning for IPC chunking and batching.

Deliverable: canary-ready backend.

## Proposed Code Touchpoints (vLLM)

- [`tmp/vllm/vllm/v1/executor/abstract.py`](tmp/vllm/vllm/v1/executor/abstract.py)
- [`tmp/vllm/vllm/config/parallel.py`](tmp/vllm/vllm/config/parallel.py)
- [`tmp/vllm/vllm/v1/executor/multiproc_executor.py`](tmp/vllm/vllm/v1/executor/multiproc_executor.py)
- new files under `tmp/vllm/vllm/v1/executor/`:
  - `sgipc_executor.py`
  - `sgipc_dispatcher.py`
  - `sgipc_bootstrap.py`

## Testing Matrix

1. **Correctness**
   - deterministic output parity vs `mp` backend.
2. **Scale**
   - TP-only, PP-only, TP+PP, DP+TP.
3. **Reliability**
   - worker crash/restart, controller crash, network partition simulation.
4. **Performance**
   - throughput/latency against `ray` and `mp` baselines.

## Rollout Strategy

1. Hidden feature flag: `VLLM_EXPERIMENTAL_SGIPC=1`.
2. Internal canary on single node.
3. Multi-node staged rollout.
4. Default remains unchanged until SLO parity is met.

## How this works with TRT-LLM

TRT-LLM has its own orchestration stack (OpenAI server, gRPC, disaggregated mode, MPI/session flow) in [`serve.py`](tmp/TensorRT-LLM/tensorrt_llm/commands/serve.py:505), including disagg and MPI worker orchestration in [`disaggregated()`](tmp/TensorRT-LLM/tensorrt_llm/commands/serve.py:973) and [`disaggregated_mpi_worker()`](tmp/TensorRT-LLM/tensorrt_llm/commands/serve.py:1045).

### Recommended interop model

1. **Control-plane federation (recommended)**
   - Keep vLLM `sgipc` and TRT-LLM orchestration separate.
   - Put a gateway/router above both (OpenAI/gRPC aware) for model routing.

2. **Disaggregated cooperation (advanced)**
   - Use a shared metadata/disagg coordinator pattern.
   - Requires explicit protocol translation for request state, finish reasons, and cache ownership.

3. **Not recommended initially**
   - In-process unification of executors (too much lifecycle and comm-stack mismatch).

### Practical integration boundaries

- **Shared**: API gateway, auth, quota/routing, observability.
- **Separate**: model executors, communicator runtimes, cache allocators, PP/TP internals.

## Risks & Mitigations

1. **Config complexity risk**
   - Mitigation: strict backend-specific validation and startup diagnostics.
2. **Multi-node bootstrap fragility**
   - Mitigation: explicit handshake protocol + bounded retries/timeouts.
3. **Behavior drift vs vLLM scheduler**
   - Mitigation: keep scheduler untouched; backend only handles orchestration.
4. **Interop ambiguity with TRT-LLM**
   - Mitigation: define clear northbound API contracts, not shared executor internals.

## Final Deliverables

1. `sgipc` executor backend in vLLM.
2. Single-node + multi-node launch docs.
3. Benchmark and resiliency report vs `ray` and `mp`.
4. Deployment guide for mixed vLLM/TRT-LLM fleet behind one routing layer.
