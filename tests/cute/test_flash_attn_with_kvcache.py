"""Contract validation tests for flash_attn_with_kvcache_func.

These tests run on CPU (no CUDA required) because they only exercise the
Python-level validation layer. Numerical parity and in-place write-back
tests live in test_flash_attn_with_kvcache_gpu.py (added in Task 5).
"""

import pytest
import torch

from flash_attn.cute import flash_attn_with_kvcache_func


def _mk(shape, dtype=torch.bfloat16, device="cpu", requires_grad=False):
    t = torch.zeros(shape, dtype=dtype, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _default_inputs(B=1, S=0, seq_new=1, H=8, D=64, max_seq=128,
                    dtype=torch.bfloat16, device="cpu"):
    q = _mk((B, seq_new, H, D), dtype, device)
    k = _mk((B, seq_new, H, D), dtype, device)
    v = _mk((B, seq_new, H, D), dtype, device)
    kc = _mk((B, max_seq, H, D), dtype, device)
    vc = _mk((B, max_seq, H, D), dtype, device)
    cs = torch.tensor([S] * B, dtype=torch.int32, device=device)
    return dict(q=q, k=k, v=v, k_cache=kc, v_cache=vc, cache_seqlens=cs)


def test_causal_true_raises():
    with pytest.raises(NotImplementedError, match="causal=False"):
        flash_attn_with_kvcache_func(**_default_inputs(), causal=True)


def test_num_splits_not_one_raises():
    with pytest.raises(NotImplementedError, match="num_splits=1"):
        flash_attn_with_kvcache_func(**_default_inputs(), num_splits=2)


def test_gqa_raises():
    inp = _default_inputs(H=32)
    inp["k"] = _mk((1, 1, 8, 64))
    inp["v"] = _mk((1, 1, 8, 64))
    with pytest.raises(NotImplementedError, match="MHA only"):
        flash_attn_with_kvcache_func(**inp)


def test_requires_grad_raises():
    inp = _default_inputs()
    inp["q"] = _mk((1, 1, 8, 64), requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        flash_attn_with_kvcache_func(**inp)


def test_dtype_mismatch_raises():
    inp = _default_inputs(dtype=torch.bfloat16)
    inp["k_cache"] = _mk(inp["k_cache"].shape, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="dtype"):
        flash_attn_with_kvcache_func(**inp)


def test_dtype_unsupported_raises():
    inp = _default_inputs(dtype=torch.float32)
    with pytest.raises(RuntimeError, match="dtype"):
        flash_attn_with_kvcache_func(**inp)


def test_shape_rank_mismatch_raises():
    inp = _default_inputs()
    inp["q"] = _mk((1, 1, 8, 64, 1))  # rank 5, not 4
    with pytest.raises(RuntimeError, match="rank"):
        flash_attn_with_kvcache_func(**inp)


def test_cache_overflow_raises():
    # max_seq=4, cache already has 3, seq_new=2 → 5 > 4
    inp = _default_inputs(max_seq=4, S=3, seq_new=2)
    inp["q"] = _mk((1, 2, 8, 64))
    inp["k"] = _mk((1, 2, 8, 64))
    inp["v"] = _mk((1, 2, 8, 64))
    with pytest.raises(RuntimeError, match="max_seq|overflow|exceed"):
        flash_attn_with_kvcache_func(**inp)


def test_stub_raises_not_implemented():
    """Until Task 5, valid inputs still raise because the kernel is not wired."""
    with pytest.raises(NotImplementedError, match="kernel wiring pending"):
        flash_attn_with_kvcache_func(**_default_inputs())


@pytest.mark.parametrize("bad_kwarg", [
    "rotary_cos", "cache_batch_idx", "cache_leftpad", "page_table",
    "window_size", "attention_chunk", "softcap", "pack_gqa",
    "cu_seqlens_q", "scheduler_metadata", "return_softmax_lse",
])
def test_out_of_scope_kwargs_raise(bad_kwarg):
    """Any FA3 kwarg outside the P0 subset must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match=bad_kwarg):
        flash_attn_with_kvcache_func(**_default_inputs(), **{bad_kwarg: object()})
