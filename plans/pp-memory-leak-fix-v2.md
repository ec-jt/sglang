# PP Memory Leak Fix - Version 2

## Problem Summary

After applying the initial fix (adding `_req_inc_lock_ref()` to all `can_run_list.append()` calls), the memory leak is still occurring:
- `req_to_token_pool memory leak detected! available_size=57, total_size=64` (7 slots leaked)
- `token_to_kv_pool_allocator memory leak detected!` (~1744 tokens leaked)

## Configuration

From `entrypoint.sh`:
- `--pp-size 8` (8 PP stages)
- `--tp-size 8` (8 TP workers per node)
- `SGLANG_DCP=8` (DCP enabled)
- `--enable-hierarchical-cache` (HiCache enabled)
- `--hicache-ratio 2`
- `--pp-async-batch-depth 0`

## Analysis

### Fix Applied

The fix added `_req_inc_lock_ref(req)` calls to 5 locations in `schedule_policy.py`:
1. `_add_dllm_req` (line 553)
2. `add_dllm_staging_req` (line 579)
3. `add_chunked_req` (line 620)
4. `add_one_req_ignore_eos` (lines 717, 734)

The `add_one_req` function already had `_req_inc_lock_ref` calls at lines 807, 812, and 846.

### Why the Fix Might Not Be Working

1. **Docker Image Not Updated**: The user may not have properly rebuilt the Docker image with the fix. Need to verify the fix is actually in the running container.

2. **Different Code Path**: The leak might be happening in a different code path that we haven't identified yet.

3. **HiCache-Specific Issue**: With `--enable-hierarchical-cache`, the `HiRadixCache` class is used instead of `RadixCache`. While `HiRadixCache` inherits from `RadixCache` and uses the same `inc_lock_ref`/`dec_lock_ref` methods, there might be HiCache-specific code paths that cause leaks.

4. **DCP-Specific Issue**: With `SGLANG_DCP=8`, the `DcpTokenToKVPoolAllocator` is used. This allocator wraps the `PagedTokenToKVPoolAllocator` and expands the page size by `dcp_world_size`. There might be DCP-specific accounting issues.

5. **PP Desync Recovery**: The PP desync handling code at lines 99-108 in `scheduler_pp_mixin.py` releases KV cache but might not properly handle all cleanup.

## Potential Root Causes

### Hypothesis 1: HiCache Load-Back Leak

In `HiRadixCache.load_back()` (line 930), `inc_lock_ref` is called:
```python
self.evictable_size_ += len(device_indices)
self.inc_lock_ref(last_hit_node)
```

But if the load-back fails or is interrupted, the lock ref might not be decremented.

### Hypothesis 2: HiCache Write-Through Leak

In `HiRadixCache.write_backup()` (line 642), `inc_lock_ref` is called:
```python
if not write_back:
    self.inc_lock_ref(node)
```

The corresponding `dec_lock_ref` is called in `writing_check()` (line 710), but if the write fails or is interrupted, the lock ref might not be decremented.

### Hypothesis 3: PP Desync Cleanup Incomplete

When PP desync is detected (lines 99-108), `release_kv_cache` is called but:
1. The `req_to_token_pool` slot might not be freed
2. The request might not be properly removed from tracking structures

### Hypothesis 4: Double Lock Ref Increment

At line 807 in `add_one_req`:
```python
self._add_dllm_req(req, prefix_len)
self._req_inc_lock_ref(req)  # <-- This is called AFTER _add_dllm_req
```

But `_add_dllm_req` already calls `_req_inc_lock_ref` at line 553. This means DLLM requests get their lock ref incremented TWICE, which would cause over-protection (not under-protection).

## Recommended Actions

### 1. Verify Fix is Applied

First, verify that the fix is actually in the running container:
```bash
docker exec <container> grep -n "_req_inc_lock_ref" /path/to/schedule_policy.py
```

### 2. Add Debug Logging

Add debug logging to track lock ref increments/decrements:
```python
def inc_lock_ref(self, node: TreeNode):
    logger.debug(f"inc_lock_ref: node={node.id}, current_lock_ref={node.lock_ref}")
    # ... existing code ...

def dec_lock_ref(self, node: TreeNode):
    logger.debug(f"dec_lock_ref: node={node.id}, current_lock_ref={node.lock_ref}")
    # ... existing code ...
```

### 3. Fix Double Lock Ref Increment

Remove the duplicate `_req_inc_lock_ref` call at line 807:
```python
if self.dllm_config is not None:
    if self.rem_dllm_tokens <= 0:
        return AddReqResult.OTHER

    self._add_dllm_req(req, prefix_len)
    # REMOVE: self._req_inc_lock_ref(req)  # Already called in _add_dllm_req
```

### 4. Add req_to_token_pool Cleanup to PP Desync Handler

In `scheduler_pp_mixin.py` lines 99-108, add req_to_token_pool cleanup:
```python
if not self.pp_group.is_first_rank and pp_proxy_tensors is None:
    logger.warning(
        f"[PP{self.pp_rank}] Skipping batch due to missing proxy tensors (PP desync detected). "
        f"Releasing KV cache for {len(self.cur_batch.reqs)} requests to prevent memory leak."
    )
    # Free KV cache and req_to_token_pool for all requests in the skipped batch
    for req in self.cur_batch.reqs:
        release_kv_cache(req, self.tree_cache, is_insert=False)
        # Also free the req_to_token_pool slot
        if req.req_pool_idx is not None:
            self.req_to_token_pool.free(req)
    self.cur_batch = None
    self.mbs[mb_id] = None
```

### 5. Check HiCache Load-Back Error Handling

Ensure that `HiRadixCache.load_back()` properly handles errors and decrements lock ref on failure.

### 6. Disable HiCache for Testing

To isolate the issue, try running without HiCache:
```bash
# Remove these lines from entrypoint.sh:
# --enable-hierarchical-cache \
# --hicache-ratio 2 \
# --hicache-write-policy write_through \
# --hicache-io-backend kernel \
# --hicache-mem-layout layer_first \
```

If the leak disappears, the issue is HiCache-specific.

### 7. Disable DCP for Testing

To isolate the issue, try running without DCP:
```bash
export SGLANG_DCP=1  # Instead of 8
```

If the leak disappears, the issue is DCP-specific.

## Next Steps

1. User should verify the fix is in the running container
2. User should try disabling HiCache and DCP to isolate the issue
3. If the issue persists, add debug logging to track lock ref changes
4. Apply the additional fixes (double lock ref, PP desync cleanup)
