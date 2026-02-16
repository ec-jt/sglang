#!/usr/bin/env python3
"""
Test script for FP4 fused decode attention kernel.

Compares the output of the fused FP4 kernel against the reference path
(software dequant + standard decode attention).

Usage:
    python -m sglang.srt.layers.attention.triton_ops.test_decode_attention_fp4
"""

import torch
import triton

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
)
from sglang.srt.layers.attention.triton_ops.decode_attention_fp4 import (
    decode_attention_fwd_fp4,
)
from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil


def test_fp4_decode_attention(
    batch_size=4,
    num_q_heads=32,
    num_kv_heads=32,
    head_dim=128,
    max_seq_len=512,
    max_kv_splits=8,
    sm_scale=None,
    device="cuda",
):
    """Test FP4 fused decode attention against reference implementation."""
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)

    torch.manual_seed(42)

    # Generate random Q
    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Generate random K, V in BF16 then quantize to FP4
    # Simulate variable sequence lengths
    seq_lens = torch.randint(64, max_seq_len + 1, (batch_size,), device=device)
    total_tokens = seq_lens.sum().item()

    # Create KV cache in BF16
    k_bf16 = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Quantize to FP4
    k_fp4, k_scale = KVFP4QuantizeUtil.batched_quantize(k_bf16)
    v_fp4, v_scale = KVFP4QuantizeUtil.batched_quantize(v_bf16)

    # Dequantize for reference
    k_deq = KVFP4QuantizeUtil.batched_dequantize(k_fp4, k_scale)
    v_deq = KVFP4QuantizeUtil.batched_dequantize(v_fp4, v_scale)

    # Build kv_indptr and kv_indices
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(total_tokens, dtype=torch.int64, device=device)

    # num_kv_splits
    num_kv_splits = torch.full((batch_size,), max_kv_splits, dtype=torch.int32, device=device)

    # Reference output using dequantized buffers
    o_ref = torch.zeros(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    attn_logits_ref = torch.zeros(
        batch_size, num_q_heads, max_kv_splits, head_dim,
        dtype=torch.float32, device=device,
    )
    attn_lse_ref = torch.zeros(
        batch_size, num_q_heads, max_kv_splits,
        dtype=torch.float32, device=device,
    )

    decode_attention_fwd(
        q, k_deq, v_deq, o_ref,
        kv_indptr, kv_indices,
        attn_logits_ref, attn_lse_ref,
        num_kv_splits, max_kv_splits,
        sm_scale,
    )

    # FP4 fused output
    o_fp4 = torch.zeros(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    attn_logits_fp4 = torch.zeros(
        batch_size, num_q_heads, max_kv_splits, head_dim,
        dtype=torch.float32, device=device,
    )
    attn_lse_fp4 = torch.zeros(
        batch_size, num_q_heads, max_kv_splits,
        dtype=torch.float32, device=device,
    )

    decode_attention_fwd_fp4(
        q, k_fp4, k_scale, v_fp4, v_scale, o_fp4,
        kv_indptr, kv_indices,
        attn_logits_fp4, attn_lse_fp4,
        num_kv_splits, max_kv_splits,
        sm_scale,
    )

    # Compare
    max_diff = (o_ref.float() - o_fp4.float()).abs().max().item()
    mean_diff = (o_ref.float() - o_fp4.float()).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.float().reshape(-1), o_fp4.float().reshape(-1), dim=0
    ).item()

    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Cos sim:   {cos_sim:.6f}")

    # FP4 has limited precision, so we use relaxed tolerances
    # The key is that both paths produce the same result from the same FP4 data
    assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim}"
    assert max_diff < 0.5, f"Max diff too high: {max_diff}"

    return True


def benchmark_fp4_decode_attention(
    batch_size=32,
    num_q_heads=32,
    num_kv_heads=32,
    head_dim=128,
    seq_len=2048,
    max_kv_splits=16,
    device="cuda",
):
    """Benchmark FP4 fused vs dequant+standard decode attention."""
    sm_scale = 1.0 / (head_dim**0.5)
    torch.manual_seed(42)

    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    total_tokens = batch_size * seq_len

    k_bf16 = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    k_fp4, k_scale = KVFP4QuantizeUtil.batched_quantize(k_bf16)
    v_fp4, v_scale = KVFP4QuantizeUtil.batched_quantize(v_bf16)

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(total_tokens, dtype=torch.int64, device=device)
    num_kv_splits = torch.full((batch_size,), max_kv_splits, dtype=torch.int32, device=device)

    # Benchmark: dequant + standard kernel
    def ref_fn():
        k_deq = KVFP4QuantizeUtil.batched_dequantize(k_fp4, k_scale)
        v_deq = KVFP4QuantizeUtil.batched_dequantize(v_fp4, v_scale)
        o = torch.zeros(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
        attn_logits = torch.zeros(batch_size, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device)
        attn_lse = torch.zeros(batch_size, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)
        decode_attention_fwd(q, k_deq, v_deq, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits, max_kv_splits, sm_scale)
        return o

    # Benchmark: fused FP4 kernel
    def fp4_fn():
        o = torch.zeros(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
        attn_logits = torch.zeros(batch_size, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device)
        attn_lse = torch.zeros(batch_size, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)
        decode_attention_fwd_fp4(q, k_fp4, k_scale, v_fp4, v_scale, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits, max_kv_splits, sm_scale)
        return o

    # Warmup
    for _ in range(5):
        ref_fn()
        fp4_fn()
    torch.cuda.synchronize()

    # Benchmark
    ref_ms = triton.testing.do_bench(ref_fn, warmup=10, rep=50)
    fp4_ms = triton.testing.do_bench(fp4_fn, warmup=10, rep=50)

    print(f"\n  Dequant + standard: {ref_ms:.3f} ms")
    print(f"  Fused FP4:          {fp4_ms:.3f} ms")
    print(f"  Speedup:            {ref_ms / fp4_ms:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: MHA (kv_group_num=1), small batch")
    print("=" * 60)
    test_fp4_decode_attention(
        batch_size=4, num_q_heads=32, num_kv_heads=32,
        head_dim=128, max_seq_len=512, max_kv_splits=8,
    )
    print("PASSED!\n")

    print("=" * 60)
    print("Test 2: GQA (kv_group_num=4), small batch")
    print("=" * 60)
    test_fp4_decode_attention(
        batch_size=4, num_q_heads=32, num_kv_heads=8,
        head_dim=128, max_seq_len=512, max_kv_splits=8,
    )
    print("PASSED!\n")

    print("=" * 60)
    print("Test 3: MHA, larger batch and seq len")
    print("=" * 60)
    test_fp4_decode_attention(
        batch_size=16, num_q_heads=32, num_kv_heads=32,
        head_dim=128, max_seq_len=2048, max_kv_splits=16,
    )
    print("PASSED!\n")

    print("=" * 60)
    print("Benchmark: MHA, batch=32, seq_len=2048")
    print("=" * 60)
    benchmark_fp4_decode_attention(
        batch_size=32, num_q_heads=32, num_kv_heads=32,
        head_dim=128, seq_len=2048, max_kv_splits=16,
    )
    print()

    print("All tests passed!")
