# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient decode attention with FP4 (E2M1) KV cache.

Reads FP4 packed data + scale factors directly from HBM and dequantizes
on-the-fly in SRAM during attention computation. This avoids the expensive
full-buffer dequantization that occurs with the standard path.

FP4 E2M1 format:
  - 4 bits per value: 1 sign bit + 2 exponent bits + 1 mantissa bit
  - Magnitude values: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] for indices 0-7
  - Block scaling: one uint8 exponent per 16 elements, scale = 2^(exp - 127)
  - Two FP4 values packed per uint8: low nibble = even index, high nibble = odd index

Buffer layouts:
  - K/V FP4: (total_tokens, num_kv_heads, head_dim // 2), dtype=uint8
  - K/V scales: (total_tokens, (num_kv_heads * head_dim) // 16), dtype=uint8
"""

import logging

import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _MIN_BLOCK_KV,
    _fwd_kernel_stage2,
    tanh,
)
from sglang.srt.utils import is_hip

_is_hip = is_hip()
logger = logging.getLogger(__name__)


@triton.jit
def _e2m1_dequant(nibble):
    """
    Dequantize a 4-bit E2M1 value (as uint8 with value 0-15) to float32.
    Sign is bit 3, magnitude index is bits 0-2.
    Magnitude lookup: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    """
    sign = ((nibble >> 3) & 1).to(tl.float32)
    mag_idx = (nibble & 0x07).to(tl.float32)

    # Piecewise lookup: idx 0-3 -> idx*0.5, idx 4->2, 5->3, 6->4, 7->6
    mag = tl.where(
        mag_idx < 4.0,
        mag_idx * 0.5,
        tl.where(
            mag_idx < 5.0,
            2.0,
            tl.where(mag_idx < 6.0, 3.0, tl.where(mag_idx < 7.0, 4.0, 6.0)),
        ),
    )
    return (1.0 - 2.0 * sign) * mag


@triton.jit
def _load_and_dequant_fp4(
    buffer_ptr,
    scale_ptr,
    token_locs,  # [BLOCK_N]
    kv_head_idx,
    stride_buf_bs,
    stride_buf_h,
    stride_scale_bs,
    head_dim: tl.constexpr,
    scale_block_size: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    valid_mask,  # [BLOCK_N] bool
):
    """
    Load FP4 packed data and scales for a block of tokens, dequantize to float32.

    Returns:
        even_vals: [BLOCK_N, HALF_DIM] float32 - values at even indices
        odd_vals: [BLOCK_N, HALF_DIM] float32 - values at odd indices
    """
    offs_half = tl.arange(0, HALF_DIM)
    mask_half = offs_half < (head_dim // 2)

    # Load packed FP4 bytes
    offs_buf = (
        token_locs[:, None] * stride_buf_bs
        + kv_head_idx * stride_buf_h
        + offs_half[None, :]
    )
    packed = tl.load(
        buffer_ptr + offs_buf,
        mask=valid_mask[:, None] & mask_half[None, :],
        other=0,
    )  # [BLOCK_N, HALF_DIM] uint8

    # Load scale factors
    # Scale layout: (total_tokens, (num_kv_heads * head_dim) // scale_block_size)
    # For head h, dim d: scale_idx = h * (head_dim // scale_block_size) + d // scale_block_size
    # For packed dim d_half: original dims are 2*d_half and 2*d_half+1
    # Both share the same scale at index (2*d_half) // scale_block_size = d_half // (scale_block_size // 2)
    scale_head_offset = kv_head_idx * (head_dim // scale_block_size)
    scale_group = offs_half // (scale_block_size // 2)

    offs_scale = (
        token_locs[:, None] * stride_scale_bs
        + (scale_head_offset + scale_group)[None, :]
    )
    scales_raw = tl.load(
        scale_ptr + offs_scale,
        mask=valid_mask[:, None] & mask_half[None, :],
        other=127,  # neutral scale (2^0 = 1.0)
    )  # [BLOCK_N, HALF_DIM] uint8

    # Unpack nibbles
    lo = packed & 0x0F  # even indices
    hi = (packed >> 4) & 0x0F  # odd indices

    # Dequantize
    even_vals = _e2m1_dequant(lo)
    odd_vals = _e2m1_dequant(hi)

    # Apply block scale
    scale = tl.exp2(scales_raw.to(tl.float32) - 127.0)
    even_vals = even_vals * scale
    odd_vals = odd_vals * scale

    return even_vals, odd_vals


@triton.jit
def _fwd_kernel_stage1_fp4_mha(
    Q,
    K_Buffer_FP4,
    K_Scale_Buffer,
    V_Buffer_FP4,
    V_Scale_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_scale_kbs,
    stride_buf_vbs,
    stride_buf_vh,
    stride_scale_vbs,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    HALF_DIM_K: tl.constexpr,
    HALF_DIM_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    xai_temperature_len: tl.constexpr,
):
    """
    Stage 1 of FP4 decode attention for MHA (kv_group_num == 1).

    Key optimization: Q is split into even/odd elements to match FP4 packing.
    QK = sum(q_even * k_even) + sum(q_odd * k_odd)
    acc_even += softmax(QK) * v_even
    acc_odd += softmax(QK) * v_odd
    Output is interleaved at store time.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    # Load Q split into even and odd elements
    offs_half_k = tl.arange(0, HALF_DIM_K)
    mask_half_k = offs_half_k < (Lk // 2)

    q_even = tl.load(
        Q + cur_batch * stride_qbs + cur_head * stride_qh + offs_half_k * 2,
        mask=mask_half_k,
        other=0.0,
    )  # [HALF_DIM_K]
    q_odd = tl.load(
        Q + cur_batch * stride_qbs + cur_head * stride_qh + offs_half_k * 2 + 1,
        mask=mask_half_k,
        other=0.0,
    )  # [HALF_DIM_K]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    # Accumulate V in even/odd halves separately
    acc_even = tl.zeros([HALF_DIM_V], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_DIM_V], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=valid_mask,
                other=0,
            )

            # Load and dequantize K
            k_even, k_odd = _load_and_dequant_fp4(
                K_Buffer_FP4,
                K_Scale_Buffer,
                kv_loc,
                cur_kv_head,
                stride_buf_kbs,
                stride_buf_kh,
                stride_scale_kbs,
                Lk,
                SCALE_BLOCK_SIZE,
                HALF_DIM_K,
                BLOCK_N,
                valid_mask,
            )  # [BLOCK_N, HALF_DIM_K] each

            # Compute QK = sum(q_even * k_even) + sum(q_odd * k_odd)
            qk = tl.sum(q_even[None, :] * k_even, 1) + tl.sum(
                q_odd[None, :] * k_odd, 1
            )  # [BLOCK_N]
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(valid_mask, qk, float("-inf"))

            # Load and dequantize V
            v_even, v_odd = _load_and_dequant_fp4(
                V_Buffer_FP4,
                V_Scale_Buffer,
                kv_loc,
                cur_kv_head,
                stride_buf_vbs,
                stride_buf_vh,
                stride_scale_vbs,
                Lv,
                SCALE_BLOCK_SIZE,
                HALF_DIM_V,
                BLOCK_N,
                valid_mask,
            )  # [BLOCK_N, HALF_DIM_V] each

            # Online softmax
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)  # [BLOCK_N]

            acc_even *= re_scale
            acc_odd *= re_scale
            acc_even += tl.sum(p[:, None] * v_even, 0)  # [HALF_DIM_V]
            acc_odd += tl.sum(p[:, None] * v_odd, 0)  # [HALF_DIM_V]

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        # Normalize
        acc_even = acc_even / e_sum
        acc_odd = acc_odd / e_sum

        # Store interleaved output: out[2i] = acc_even[i], out[2i+1] = acc_odd[i]
        offs_dv_half = tl.arange(0, HALF_DIM_V)
        mask_dv_half = offs_dv_half < (Lv // 2)

        offs_mid_o_base = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        )

        # Store even indices
        tl.store(
            Att_Out + offs_mid_o_base + offs_dv_half * 2,
            acc_even,
            mask=mask_dv_half,
        )
        # Store odd indices
        tl.store(
            Att_Out + offs_mid_o_base + offs_dv_half * 2 + 1,
            acc_odd,
            mask=mask_dv_half,
        )

        # Store LSE
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


# ============================================================================
# Python wrapper functions
# ============================================================================


def _decode_att_m_fwd_fp4(
    q,
    k_buffer_fp4,
    k_scale_buffer,
    v_buffer_fp4,
    v_scale_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    """Launch FP4 decode attention stage 1 for MHA."""
    BLOCK = 64
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = q.shape[-1]  # Q head dim (full, not packed)
    Lv = Lk  # For MHA, V head dim == K head dim

    batch, head_num = q.shape[0], q.shape[1]
    num_kv_heads = k_buffer_fp4.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = head_num // num_kv_heads

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    HALF_DIM_K = triton.next_power_of_2(Lk // 2)
    HALF_DIM_V = triton.next_power_of_2(Lv // 2)

    _fwd_kernel_stage1_fp4_mha[grid](
        q,
        k_buffer_fp4,
        k_scale_buffer,
        v_buffer_fp4,
        v_scale_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer_fp4.stride(0),
        k_buffer_fp4.stride(1),
        k_scale_buffer.stride(0),
        v_buffer_fp4.stride(0),
        v_buffer_fp4.stride(1),
        v_scale_buffer.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        HALF_DIM_K=HALF_DIM_K,
        HALF_DIM_V=HALF_DIM_V,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        Lk=Lk,
        Lv=Lv,
        SCALE_BLOCK_SIZE=16,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
    )


def decode_attention_fwd_fp4(
    q,
    k_buffer_fp4,
    k_scale_buffer,
    v_buffer_fp4,
    v_scale_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    """
    Decode attention with FP4 KV cache.

    Args:
        q: [batch, num_q_heads, head_dim] float16/bfloat16
        k_buffer_fp4: [total_tokens, num_kv_heads, head_dim // 2] uint8 (packed FP4)
        k_scale_buffer: [total_tokens, (num_kv_heads * head_dim) // 16] uint8
        v_buffer_fp4: [total_tokens, num_kv_heads, head_dim // 2] uint8 (packed FP4)
        v_scale_buffer: [total_tokens, (num_kv_heads * head_dim) // 16] uint8
        o: [batch, num_q_heads, v_head_dim] output tensor
        kv_indptr: [batch + 1] int32
        kv_indices: [total_kv_tokens] int32
        attn_logits: [batch, num_q_heads, max_kv_splits, v_head_dim] scratch buffer
        attn_lse: [batch, num_q_heads, max_kv_splits] scratch buffer
        num_kv_splits: [batch] int32
        max_kv_splits: int
        sm_scale: float
        logit_cap: float
    """
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    # Stage 1: Compute attention scores and weighted V sums per split
    _decode_att_m_fwd_fp4(
        q,
        k_buffer_fp4,
        k_scale_buffer,
        v_buffer_fp4,
        v_scale_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
        xai_temperature_len,
    )

    # Stage 2: Reduce across splits (reuse existing stage2 kernel)
    _decode_softmax_reducev_fwd_fp4(
        attn_logits,
        attn_lse,
        q,
        o,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        q.shape[-1],  # Lv = Lk for MHA
        sinks,
    )


def _decode_softmax_reducev_fwd_fp4(
    logits,
    lse,
    q,
    o,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
    Lv,
    sinks=None,
):
    """Stage 2: reduce across KV splits. Reuses the standard stage2 kernel."""
    batch, head_num = q.shape[0], q.shape[1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        kv_indptr,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=HAS_SINK,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )
