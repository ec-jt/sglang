# PP Memory Leak Root Cause Analysis

## Status: Root Cause Found

**Date:** 2026-02-26

## Problem

After running benchmark 1 (100 prompts, sharegpt), consistently:
- `req_to_token_pool memory leak detected! available_size=57, total_size=64` → **7 slots leaked**
- `token_to_kv_pool_allocator memory leak detected!` → **~2072-2136 tokens leaked**

The leak is **deterministic** - always 7 slots, across multiple runs.

## Root Cause: PP Event Loop Batch Result Processing Race

In `scheduler_pp_mixin.py` `event_loop_pp`, the inner loop processes microbatch results with a 1-slot delay:

```python
for mb_id in range(pp_loop_size):  # 0..7
    next_mb_id = (mb_id + 1) % pp_loop_size
    
    # Line 91: OVERWRITES self.mbs[mb_id] with new batch or None
    self.mbs[mb_id] = self.get_next_batch_to_run()
    
    # ... launch batch ...
    
    # Line 137: Processes result for self.mbs[next_mb_id]
    if self.mbs[next_mb_id] is not None:
        self._pp_process_batch_result(self.mbs[next_mb_id], ...)
```

### The Race Condition

When transitioning from busy to idle:

```
Busy iteration (last):
  mb_id=0: self.mbs[0] = batch_A (7 requests)  ← launched
  mb_id=1: self.mbs[1] = batch_B               ← launched
  ...

Idle iteration (next):
  mb_id=0: self.mbs[0] = None  ← OVERWRITES batch_A before it can be processed!
           processes self.mbs[1] = batch_B  ✓ (still has old value)
  mb_id=1: self.mbs[1] = None
           processes self.mbs[2]  ✓
  ...
  mb_id=6: self.mbs[6] = None
           processes self.mbs[7]  ✓
  mb_id=7: self.mbs[7] = None
           processes self.mbs[0] = None  ✗ ALREADY OVERWRITTEN!
```

**Result**: `batch_A` in `self.mbs[0]` is overwritten at `mb_id=0` before `mb_id=7` can process it. The 7 requests in `batch_A` never have `release_kv_cache` called, leaking both req_to_token_pool slots and KV cache tokens.

### Why Always 7

The last decode batch before idle had 7 requests still running. These were in `self.mbs[0]` (the first microbatch slot), and their results were lost when the slot was overwritten during the idle transition.

## Fix

Save the old batch before overwriting, and process it if it hasn't been processed yet. Or add a drain loop before `self_check_during_idle()`.

### Option A: Save and Process Before Overwrite

```python
# Before overwriting self.mbs[mb_id], save the old batch
old_batch = self.mbs[mb_id]
self.mbs[mb_id] = self.get_next_batch_to_run()

# ... later, when processing next_mb_id ...
# If old_batch was not processed by the next_mb_id logic, process it now
```

### Option B: Drain Loop Before Idle Check (Simpler)

```python
if server_is_idle:
    # Drain any remaining unprocessed batches
    for i in range(self.pp_loop_size):
        if self.mbs[i] is not None:
            # Process remaining batch results
            ...
    self.self_check_during_idle()
```

## All Fixes Applied in This Session

| # | File | Fix | Purpose |
|---|------|-----|---------|
| 1 | `schedule_policy.py` | Added `_req_inc_lock_ref` to 5 locations | Memory accounting for cached prefixes |
| 2 | `schedule_policy.py` | Removed duplicate `_req_inc_lock_ref` at line 807 | Bug fix for DLLM double lock |
| 3 | `scheduler_pp_mixin.py` | Added `release_kv_cache` + `req_to_token_pool.free` to 3 PP desync handlers | PP desync cleanup |
| 4 | `parallel_state.py` | Changed `recv_object`/`recv_tensor_dict` to return None on timeout | Graceful PP recovery |
| 5 | `common.py` | Changed `broadcast_pyobj` to return None on timeout | Graceful PP recovery |
| 6 | `detokenizer_manager.py` | Non-blocking ZMQ send with fallback | Prevent detokenizer timeout |
| 7 | `scheduler_pp_mixin.py` | **TODO**: Fix batch result processing race | **ROOT CAUSE of 7-slot leak** |
