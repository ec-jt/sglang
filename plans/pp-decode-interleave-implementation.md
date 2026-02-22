# PP Decode Interleave Implementation Plan

## Problem

With PP=4 and chunked prefill, the scheduler in [`get_next_batch_to_run()`](python/sglang/srt/managers/scheduler.py:1880) **always prioritizes prefill over decode** (line 1942-1944). Since `--enable-mixed-chunk` is incompatible with PP (assertion at [`server_args.py:5107`](python/sglang/srt/server_args.py:5107)), decode batches are completely starved — 70+ consecutive prefill batches run before any decode batch, causing 1.75 tok/s generation throughput.

## Solution

Add a `pp_decode_interleave_interval` parameter that forces the scheduler to run a decode batch after every N consecutive prefill batches when PP > 1. This is safe because the chunked prefill stash/restore mechanism already handles interruptions.

## Detailed Changes

### Change 1: Add dataclass field to ServerArgs

**File**: [`python/sglang/srt/server_args.py`](python/sglang/srt/server_args.py:351)  
**Location**: Line 351, after `pp_async_batch_depth`  
**What**: Add new field `pp_decode_interleave_interval: int = 0`  
**Why**: Default 0 means disabled (no behavior change for existing users). When set to e.g. 4, forces decode every 4 prefill steps.

```python
# Before (line 350-352):
    pp_max_micro_batch_size: Optional[int] = None
    pp_async_batch_depth: int = 0
    stream_interval: int = 1

# After:
    pp_max_micro_batch_size: Optional[int] = None
    pp_async_batch_depth: int = 0
    pp_decode_interleave_interval: int = 0
    stream_interval: int = 1
```

### Change 2: Add CLI argument

**File**: [`python/sglang/srt/server_args.py`](python/sglang/srt/server_args.py:3319)  
**Location**: Line 3319, after the `--pp-async-batch-depth` argument block  
**What**: Add argparse argument for `--pp-decode-interleave-interval`

```python
# After the pp-async-batch-depth block (line 3319), insert:
        parser.add_argument(
            "--pp-decode-interleave-interval",
            type=int,
            default=ServerArgs.pp_decode_interleave_interval,
            help="Force a decode batch after this many consecutive prefill batches in PP mode. "
            "0 means disabled (default). Recommended value: 4. "
            "This prevents decode starvation when using pipeline parallelism with chunked prefill.",
        )
```

### Change 3: Initialize counter in scheduler

**File**: [`python/sglang/srt/managers/scheduler.py`](python/sglang/srt/managers/scheduler.py:766)  
**Location**: Line 766, at the end of `init_running_status()`  
**What**: Add `self.pp_prefill_steps_since_decode = 0` counter

```python
# Before (line 764-767):
        self.sessions: Dict[str, Session] = {}
        self.forward_sleep_time = None
        self._engine_paused = False

# After:
        self.sessions: Dict[str, Session] = {}
        self.forward_sleep_time = None
        self._engine_paused = False
        self.pp_prefill_steps_since_decode = 0
```

### Change 4: Add interleaving logic in get_next_batch_to_run

**File**: [`python/sglang/srt/managers/scheduler.py`](python/sglang/srt/managers/scheduler.py:1929)  
**Location**: Line 1929-1951, the scheduling decision block  
**What**: After `get_new_batch_prefill()` returns a batch, check if we should force decode instead

**Current code** (lines 1926-1951):
```python
        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm()
        else:
            new_batch = self.get_new_batch_prefill()

        need_mlp_sync = self.require_mlp_sync
        if need_mlp_sync and not self.spec_algorithm.is_none():
            new_batch = self.maybe_prepare_mlp_sync_batch_and_log_stats(
                new_batch, log_stats=False
            )
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode
            if not self.running_batch.is_empty():
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None
```

**New code** (replace lines 1942-1951):
```python
        # PP decode interleaving: force decode after N consecutive prefills
        # to prevent decode starvation in pipeline parallelism mode
        pp_interleave = self.server_args.pp_decode_interleave_interval
        if (
            pp_interleave > 0
            and new_batch is not None
            and not self.running_batch.is_empty()
            and self.pp_prefill_steps_since_decode >= pp_interleave
        ):
            new_batch = None  # Suppress prefill, force decode this step

        if new_batch is not None:
            # Run prefill first if possible
            self.pp_prefill_steps_since_decode += 1
            ret = new_batch
        else:
            # Run decode
            self.pp_prefill_steps_since_decode = 0
            if not self.running_batch.is_empty():
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None
```

**Key design decisions**:
1. The interleaving check happens AFTER `get_new_batch_prefill()` returns — we don't prevent the prefill batch from being created, we just suppress it. This means the `chunked_req` was already stashed at line 1898 and will be restored on the next cycle automatically.
2. We only force decode when `running_batch` is not empty — if there are no decode requests, there's nothing to decode.
3. The counter resets to 0 on ANY non-prefill step (decode or idle), not just forced decode.
4. Setting `new_batch = None` causes the existing decode path to execute naturally — no new code paths.

### Change 5: Update entrypoint

**File**: [`../entrypoint.sh`](../entrypoint.sh:105)  
**Location**: Line 105, after `--page-size`  
**What**: Add `--pp-decode-interleave-interval 4` and change `--page-size` from 64 to 16

```bash
# Before (line 105):
    --page-size 64 \

# After:
    --page-size 16 \
    --pp-decode-interleave-interval 4 \
```

## Safety Analysis

### Why suppressing new_batch is safe

When we set `new_batch = None` at the interleaving check:

1. **The chunked_req was already stashed** at line 1894-1898:
   ```python
   if self.chunked_req is not None:
       chunked_req_to_exclude.add(self.chunked_req)
       self.stash_chunked_request(self.chunked_req)
   ```
   This happens BEFORE `get_new_batch_prefill()` is called. The stash saves the chunked request's KV cache state to the tree cache.

2. **get_new_batch_prefill() restores it** at line 2054-2058:
   ```python
   if self.chunked_req is not None:
       self.chunked_req.init_next_round_input()
       self.chunked_req = adder.add_chunked_req(...)
   ```
   On the next call to `get_next_batch_to_run()`, the chunked_req will be stashed again (line 1894-1898), then `get_new_batch_prefill()` will pick it up again.

3. **The prefill batch we suppress is never used** — it was created by `get_new_batch_prefill()` but since we set `new_batch = None`, it's simply discarded. The requests in `can_run_list` were already removed from `waiting_queue` (line 2139-2141), but they'll be re-added on the next cycle because the `chunked_req` mechanism handles this.

4. **Wait — there's a subtlety**: `get_new_batch_prefill()` modifies `self.waiting_queue` by removing requests that were added to the batch. If we suppress the batch, those requests are lost!

### Correction: Move the check BEFORE get_new_batch_prefill

To avoid the issue of lost requests, the interleaving check should happen **before** calling `get_new_batch_prefill()`:

**Revised Change 4** (replace lines 1926-1951):
```python
        # PP decode interleaving: force decode after N consecutive prefills
        pp_interleave = self.server_args.pp_decode_interleave_interval
        force_decode = (
            pp_interleave > 0
            and not self.running_batch.is_empty()
            and self.pp_prefill_steps_since_decode >= pp_interleave
        )

        if force_decode:
            new_batch = None
        elif self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm()
        else:
            new_batch = self.get_new_batch_prefill()

        need_mlp_sync = self.require_mlp_sync
        if need_mlp_sync and not self.spec_algorithm.is_none():
            new_batch = self.maybe_prepare_mlp_sync_batch_and_log_stats(
                new_batch, log_stats=False
            )
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            self.pp_prefill_steps_since_decode += 1
            ret = new_batch
        else:
            # Run decode (or idle)
            self.pp_prefill_steps_since_decode = 0
            if not self.running_batch.is_empty():
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None
```

This is cleaner and avoids the request-loss issue entirely. When `force_decode` is true, we skip `get_new_batch_prefill()` completely, so no requests are removed from the waiting queue.

## Interaction with Existing Mechanisms

| Mechanism | Interaction | Safe? |
|-----------|-------------|-------|
| **Chunked prefill stash/restore** | Chunked req is stashed at line 1894-1898 before our check. On forced decode, it stays stashed and is restored next cycle. | ✅ Yes |
| **PP microbatch slots** | Each microbatch slot has its own `running_mbs[mb_id]`. The counter is global across all slots. This means decode is forced every N total prefill steps, not per-slot. | ✅ Yes |
| **MLP sync for DP attention** | The `maybe_prepare_mlp_sync_batch_and_log_stats` call still happens after our check. | ✅ Yes |
| **DLLM** | We skip DLLM batch creation too when forcing decode. DLLM is not used in this deployment. | ✅ Yes |
| **Radix cache** | No interaction — radix cache operates at the request level, not the batch scheduling level. | ✅ Yes |

## Expected Behavior

With `--pp-decode-interleave-interval 4`:

```
Step 1: Prefill (counter=1)
Step 2: Prefill (counter=2)  
Step 3: Prefill (counter=3)
Step 4: Prefill (counter=4)
Step 5: force_decode=true → Decode (counter=0)
Step 6: Prefill (counter=1)
Step 7: Prefill (counter=2)
Step 8: Prefill (counter=3)
Step 9: Prefill (counter=4)
Step 10: force_decode=true → Decode (counter=0)
...
```

With PP=4, `pp_loop_size = pp_size + pp_async_batch_depth = 4 + 0 = 4`. Each outer loop iteration processes 4 microbatch slots. So with interval=4, decode runs roughly every outer loop iteration.

## Files Modified Summary

| # | File | Lines Changed | Description |
|---|------|--------------|-------------|
| 1 | [`python/sglang/srt/server_args.py`](python/sglang/srt/server_args.py:351) | +1 line at 351 | Add `pp_decode_interleave_interval` dataclass field |
| 2 | [`python/sglang/srt/server_args.py`](python/sglang/srt/server_args.py:3319) | +6 lines at 3319 | Add CLI argument |
| 3 | [`python/sglang/srt/managers/scheduler.py`](python/sglang/srt/managers/scheduler.py:766) | +1 line at 766 | Initialize counter |
| 4 | [`python/sglang/srt/managers/scheduler.py`](python/sglang/srt/managers/scheduler.py:1926) | ~15 lines modified at 1926-1951 | Add interleaving logic |
| 5 | [`../entrypoint.sh`](../entrypoint.sh:105) | 2 lines modified at 105 | Update page-size and add new flag |
