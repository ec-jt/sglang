# PP Queue-Based Finalization Implementation Plan

## Objective

Eliminate PP lifecycle leaks while preserving throughput by moving SGLang PP completion to a queue-based model (similar lifecycle guarantees to vLLM/TensorRT-LLM) and consolidating request resource release into one termination path.

Primary target symptoms are the leak checks in [`check_memory()`](sglang/python/sglang/srt/managers/scheduler_runtime_checker_mixin.py:234) and [`_check_req_pool()`](sglang/python/sglang/srt/managers/scheduler_runtime_checker_mixin.py:213), currently triggered from PP workloads.

---

## Design Principles

1. **Single ownership of finalization**
   - Every launched PP microbatch must be finalized exactly once.
   - Resource free operations (`KV` + `req_pool`) happen only through one helper path.

2. **Schedule/launch separated from finalize/free**
   - Do not rely on mutable microbatch slot arrays as source-of-truth for completion state.
   - Track launched work in explicit queue entries until finalize succeeds.

3. **Fail-safe under desync/timeout paths**
   - PP desync should mark/route entries for centralized finalization, not inline free logic duplicated in loop branches.

4. **Performance-aware rollout**
   - Stabilize at conservative overlap first.
   - Re-enable overlap depth after invariants and perf gates pass.

---

## Scope

### In Scope
- Refactor PP event-loop completion bookkeeping in [`event_loop_pp()`](sglang/python/sglang/srt/managers/scheduler_pp_mixin.py:49), [`event_loop_pp_disagg_prefill()`](sglang/python/sglang/srt/managers/scheduler_pp_mixin.py:173), and decode-side PP loop sections in [`scheduler_pp_mixin.py`](sglang/python/sglang/srt/managers/scheduler_pp_mixin.py).
- Introduce explicit in-flight completion queue for launched PP work.
- Centralize request finalization/release helper (KV + req slot + state transitions).
- Keep strict memory checks as rollout gate.

### Out of Scope
- Major scheduler policy redesign in [`schedule_policy.py`](sglang/python/sglang/srt/managers/schedule_policy.py).
- Model execution kernel changes.

---

## Implementation Plan (Phased)

## Phase 0 — Baseline and Instrumentation

### Changes
- Add lightweight counters in PP scheduler mixin:
  - `pp_launched_batches`
  - `pp_finalized_batches`
  - `pp_launched_reqs`
  - `pp_finalized_reqs`
  - `pp_dropped_batches`
- Emit counters periodically from scheduler logs (same cadence as existing metrics path).

### Files
- [`scheduler_pp_mixin.py`](sglang/python/sglang/srt/managers/scheduler_pp_mixin.py)
- (Optional) metrics mixin in [`scheduler_metrics_mixin.py`](sglang/python/sglang/srt/managers/scheduler_metrics_mixin.py:69)

### Exit Criteria
- Counters visible per rank (`PPx/TPy`).
- No behavior change yet.

---

## Phase 1 — Add Explicit PP In-Flight Completion Queue

### Changes
- Introduce queue entry object (dataclass/tuple), e.g.:
  - `mb_id`
  - `batch_ref`
  - `result_ref`
  - `d2h_event`
  - `launch_seq`
  - `status` (`LAUNCHED`, `DESYNC_DROPPED`, `FINALIZED`)
- On successful launch, enqueue entry immediately.
- Replace direct reliance on `self.mbs[next_mb_id]` overwrite-sensitive state with queue pop/finalize flow.
- Add `drain_completed_pp_entries()` called each iteration and once before idle checks.

### Files
- [`scheduler_pp_mixin.py`](sglang/python/sglang/srt/managers/scheduler_pp_mixin.py:80)

### Exit Criteria
- `pp_finalized_batches == pp_launched_batches` over steady-state + idle transitions.
- No regression in correctness tests.

---

## Phase 2 — Single Termination/Free Path

### Changes
- Add a single helper, e.g. `_finalize_request_resources(req, reason, is_insert=False)`:
  - Executes [`release_kv_cache(...)`](sglang/python/sglang/srt/managers/scheduler_pp_mixin.py:111) (or equivalent centralized utility).
  - Frees request pool slot exactly once when `req.req_pool_idx` is live.
  - Handles idempotency guard to prevent double-free.
- Replace inline free snippets in PP desync branches with calls into helper.
- Ensure normal completion, abort, and desync all use same helper path.

### Files
- [`scheduler_pp_mixin.py`](sglang/python/sglang/srt/managers/scheduler_pp_mixin.py:104)
- [`common.py` / release utility](sglang/python/sglang/srt/mem_cache/common.py:466)
- Potential call-site alignment in [`scheduler.py`](sglang/python/sglang/srt/managers/scheduler.py)

### Exit Criteria
- No duplicated inline free logic in PP loops.
- Leak checks stay green under desync injection.

---

## Phase 3 — Desync/Timeout Integration

### Changes
- Convert “skip batch” behavior into queue state transitions:
  - mark as `DESYNC_DROPPED`
  - enqueue for centralized finalize
- Ensure no path leaves launched work untracked.
- Add explicit log with launch sequence and mb_id for dropped/finalized entries.

### Files
- [`scheduler_pp_mixin.py`](sglang/python/sglang/srt/managers/scheduler_pp_mixin.py:100)
- PP comm timeout helpers in distributed utils if needed.

### Exit Criteria
- In forced proxy-missing scenarios, finalized counters still match launched counters.

---

## Phase 4 — Performance Re-tuning

### Changes
- Validate at conservative overlap first (`pp_async_batch_depth=0`).
- Incrementally restore overlap (`1`, then tuned value) once invariants hold.
- Optional micro-optimizations:
  - reduce synchronization points in finalize path
  - batch free operations when safe

### Exit Criteria
- Throughput/latency near or above pre-fix baseline with zero leak warnings.

---

## Validation Matrix

## A. Functional
- Single-node PP smoke.
- Multi-node PP smoke.
- Chunked prefill + decode interleave.
- Health-check traffic + benchmark mixed load.

## B. Fault Injection
- Proxy tensor missing on non-first PP rank.
- Slow/timeout on inter-stage communication.
- Mid-flight request cancel/abort.

## C. Invariants (must pass)
- [`check_memory()`](sglang/python/sglang/srt/managers/scheduler_runtime_checker_mixin.py:234) clean during/after load.
- [`_check_req_pool()`](sglang/python/sglang/srt/managers/scheduler_runtime_checker_mixin.py:213) returns full pool on idle.
- `launched == finalized` for both batches and requests.

## D. Performance Gates
- No >3% throughput regression at `pp_async_batch_depth=0`.
- After overlap restore, throughput recovers to baseline band (target ±2%).
- No significant TTFT regression for long-prefill workloads.

---

## Rollout Strategy

1. Land Phase 0–2 behind feature flag (example: `SGLANG_PP_QUEUE_FINALIZE=1`).
2. Enable in canary environment only.
3. Run 24h stress with leak checks enabled.
4. Enable by default after canary passes; keep fallback path one release cycle.

---

## Rollback Plan

- Immediate rollback: disable new path via feature flag and revert to prior PP loop behavior.
- Keep added counters/logging even on rollback to preserve diagnostics.

---

## Suggested Execution Order (Task Checklist)

- [ ] Implement PP counters and logs.
- [ ] Add in-flight queue entry structure.
- [ ] Wire launch enqueue path in all PP loops.
- [ ] Implement `drain_completed_pp_entries()` and call before idle check.
- [ ] Add unified request finalization helper.
- [ ] Replace inline desync frees with unified helper.
- [ ] Add fault-injection tests for desync and abort.
- [ ] Run leak + perf validation matrix.
- [ ] Tune overlap depth and finalize rollout defaults.

