# FP4/FP8 KV Cache Decode Performance Analysis

## Problem
Both Kimi K2.5 and MiniMax M2.5 show ~50% throughput drop when using `--kv-cache-dtype fp4_e2m1`. FP8 KV cache also shows the same slowdown.

## Root Cause Summary

There are **two separate bottlenecks** depending on the model type:

### 1. FP4 KV Cache: Full Software Dequantization (Both Models)

For FP4, [`MLATokenToKVPoolFP4.get_key_buffer()`](python/sglang/srt/mem_cache/memory_pool.py:1640) and [`MHATokenToKVPoolFP4._get_key_buffer()`](python/sglang/srt/mem_cache/memory_pool.py:1097) call [`KVFP4QuantizeUtil.batched_dequantize()`](python/sglang/srt/layers/quantization/kvfp4_tensor.py:74) which dequantizes the **entire KV cache buffer** from FP4 → BF16 on every decode step, per layer.

This `@torch.compile`-decorated function does:
- Unpack uint8 → two FP4 nibbles (bit operations)
- E2M1 table lookup for magnitude values
- Sign bit application
- Block-wise scale factor multiplication (per 16-element block)
- Cast to BF16

### 2. FP8 KV Cache with FlashInfer MLA: `.to(q.dtype)` Cast (Kimi K2.5)

For FP8 with the `flashinfer` MLA backend, [`get_key_buffer()`](python/sglang/srt/mem_cache/memory_pool.py:1468) returns a zero-cost FP8 view. But then in [`forward_decode()`](python/sglang/srt/layers/attention/flashinfer_mla_backend.py:639):

```python
k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
    q.dtype  # <-- This casts the ENTIRE KV cache FP8 → BF16
)
```

This `.to(q.dtype)` creates a full copy of the entire KV cache buffer, converting FP8 → BF16, every decode step, every layer.

### 3. FP8 KV Cache with Triton MHA: Same Issue (MiniMax M2.5)

The triton backend at [`forward_decode()`](python/sglang/srt/layers/attention/triton_backend.py:1031) passes `get_key_buffer()` directly to the triton kernel. The triton kernel may handle FP8 natively, but the `MHATokenToKVPoolFP4` dequant still applies for FP4.

## Backend Comparison Matrix

| Backend | Model Type | FP4 KV | FP8 KV | BF16 KV |
|---------|-----------|--------|--------|---------|
| `flashinfer` | MLA (Kimi K2.5) | ❌ Software dequant | ❌ `.to()` cast | ✅ Native |
| `trtllm_mla` | MLA (Kimi K2.5) | ❌ Software dequant | ✅ **Native FP8** | ✅ Native |
| `cutlass_mla` | MLA (Kimi K2.5) | ❌ Software dequant | ? | ✅ Native |
| `flashmla` | MLA (Kimi K2.5) | ❌ Software dequant | ? | ✅ Native |
| `triton` | MHA (MiniMax M2.5) | ❌ Software dequant | ⚠️ Triton may handle | ✅ Native |

**Key finding**: `trtllm_mla` is the ONLY backend that can consume FP8 KV cache natively for MLA models. It calls [`get_key_buffer()`](python/sglang/srt/layers/attention/trtllm_mla_backend.py:815) without `.to()` and passes the buffer directly to `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`.

## Recommendations

### For Kimi K2.5 (MLA model)

**Option A: Best performance with memory savings** — Use `trtllm_mla` + FP8:
```bash
--attention-backend trtllm_mla \
--kv-cache-dtype fp8_e4m3 \
--page-size 64
```
The TRT-LLM MLA kernel natively consumes FP8 KV cache — no dequantization overhead. This gives 2x memory savings with near-zero decode overhead. Page size must be 32 or 64 for trtllm_mla.

**Option B: Maximum performance** — Use BF16 KV cache:
```bash
--attention-backend flashinfer \
--page-size 64
```
Remove `--kv-cache-dtype` entirely. Full speed, but uses 2x more KV cache memory.

**Option C: If you must use FP4** — No good option exists today. All backends dequantize the entire buffer. This would require a custom fused FP4-dequant+attention kernel (significant engineering effort).

### For MiniMax M2.5 (MHA model)

**Option A: Maximum performance** — Use BF16 KV cache:
```bash
--prefill-attention-backend triton \
--decode-attention-backend triton
```
Remove `--kv-cache-dtype fp4_e2m1`. Full speed.

**Option B: Memory savings with FP8** — Try `trtllm_mha`:
```bash
--decode-attention-backend trtllm_mha \
--kv-cache-dtype fp8_e4m3
```
The TRT-LLM MHA backend may have native FP8 support (needs testing).

**Option C: If you must use FP4** — Same as Kimi K2.5, no good option exists.

## Memory Budget Analysis (RTX 5090, 32GB × 8 GPUs)

For Kimi K2.5 with TP=8, PP=4, 4 nodes:
- Model weights (FP8 quantized): ~30GB per GPU (256B params / 32 GPUs)
- With `--mem-fraction-static 0.9`: ~28.8GB available per GPU
- KV cache budget: ~28.8GB - 30GB = **negative** — this is why you need FP4/FP8!

Wait — with PP=4 across 4 nodes, each node has 8 GPUs with TP=8. So each GPU holds 1/8 of the model layers (PP splits across nodes, TP splits within node). The model is ~256B params FP8 = ~256GB total. With 32 GPUs: ~8GB per GPU for weights. That leaves ~20GB per GPU for KV cache.

For MLA KV cache (kv_lora_rank=512 + qk_rope_head_dim=64 = 576 dims, 1 head):
- BF16: 576 × 2 bytes = 1,152 bytes per token per layer
- FP8: 576 × 1 byte = 576 bytes per token per layer  
- FP4: 576 / 2 = 288 bytes per token per layer (+ scales)

With ~15 layers per GPU (61 layers / 4 PP stages ≈ 15):
- BF16: 1,152 × 15 = 17,280 bytes per token = ~17KB per token
- FP8: 576 × 15 = 8,640 bytes per token = ~8.6KB per token
- FP4: ~324 × 15 = 4,860 bytes per token = ~4.9KB per token

With 20GB KV cache budget:
- BF16: ~1.2M tokens total → ~100 concurrent 12K-context requests
- FP8: ~2.4M tokens total → ~200 concurrent 12K-context requests
- FP4: ~4.3M tokens total → ~350 concurrent 12K-context requests

**FP8 with `trtllm_mla` is the sweet spot**: 2x memory savings with native kernel support (no decode overhead).

## Proposed Entrypoint Changes

### Kimi K2.5 entrypoint.sh
```diff
-    --attention-backend flashinfer \
-    --kv-cache-dtype fp4_e2m1 \
+    --attention-backend trtllm_mla \
+    --kv-cache-dtype fp8_e4m3 \
```

### MiniMax M2.5 entrypoint.sh
```diff
-    --kv-cache-dtype fp4_e2m1 \
+    # Remove FP4 KV cache - use BF16 for full speed
+    # Or try: --kv-cache-dtype fp8_e4m3 --decode-attention-backend trtllm_mha
```

## Why FP4 KV Cache Cannot Be Fast Today

No attention kernel (FlashInfer, Triton, CUTLASS, FlashMLA, TRT-LLM) supports FP4 KV cache as a native input format. The FP4 quantization in sglang is implemented entirely in Python/PyTorch via [`KVFP4QuantizeUtil`](python/sglang/srt/layers/quantization/kvfp4_tensor.py:28), with `@torch.compile` for some optimization. But the fundamental issue is:

1. **Store path**: Quantize BF16 → FP4 (runs once per new token — acceptable)
2. **Load path**: Dequantize FP4 → BF16 for the **entire buffer** (runs every decode step — unacceptable)

A proper fix would require:
- Fused FP4-dequant + attention CUDA kernels (read FP4, dequant on-the-fly in SRAM)
- Or FlashInfer/TRT-LLM adding native FP4 KV cache support
- This is a significant upstream engineering effort
