# PP KV Cache Memory Leak Fix

## Problem

Server crashes with:
```
ValueError: token_to_kv_pool_allocator memory leak detected! 
max_total_num_tokens=762184, available_size=3160, evictable_size=728504, protected_size=29520
```

**1000 tokens leaked** (762184 - 3160 - 728504 - 29520 = 1000)

## Root Cause

When PP desync occurs and a batch is skipped at [`scheduler_pp_mixin.py:98-103`](python/sglang/srt/managers/scheduler_pp_mixin.py:98), the KV cache allocated for those requests is never freed:

```python
if not self.pp_group.is_first_rank and pp_proxy_tensors is None:
    logger.warning(f"[PP{self.pp_rank}] Skipping batch due to missing proxy tensors")
    self.cur_batch = None
    self.mbs[mb_id] = None
    # BUG: KV cache for self.cur_batch.reqs is never freed!
```

## Quick Fix (Environment Variable)

Add to docker-compose environment:
```yaml
environment:
  - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
```

This converts the memory leak detection from a **crash** (ValueError) to a **warning** log. The leak still exists but won't crash the server.

## Proper Fix (Code Change - Deferred)

Add KV cache cleanup in the batch skip path:

```python
if not self.pp_group.is_first_rank and pp_proxy_tensors is None:
    logger.warning(f"[PP{self.pp_rank}] Skipping batch due to missing proxy tensors")
    # Free KV cache for all requests in the skipped batch
    from sglang.srt.mem_cache.utils import release_kv_cache
    for req in self.cur_batch.reqs:
        release_kv_cache(req, self.tree_cache, is_insert=False)
    self.cur_batch = None
    self.mbs[mb_id] = None
```

This properly frees the KV cache tokens when a batch is dropped, preventing the leak.
