# fa4-kvcache-non-paged

FA4 (`flash_attn.cute`) drop-in of FA3's `flash_attn_with_kvcache`, targeting
HRM inference on SM12 dev + SM100 prod.

- Base: `Dao-AILab/flash-attention:main` @ `2409214a03797b168f648ea30df1adbc09ce658a`
- Design doc: JoyBrain repo `docs/superpowers/specs/2026-07-22-fa4-flash-attn-with-kvcache-design.md`
- Scope: contiguous (non-paged) KV, `causal=False`, MHA, inference-only,
  `num_splits=1`, 9-kwarg API subset.
- Not for upstream. Fork-only.

## Status: **PAUSED** as of 2026-07-22

Landed on this branch:
- Task 1 (`305a7cf`) — branch bootstrap
- Task 2 (`4306ac6`) — `flash_attn_with_kvcache_func` interface skeleton with full P0
  validation + 20 contract-raise tests (CPU-only). The body still raises
  `NotImplementedError("kernel wiring pending; will land in Task 5")`.
- Task 3a (`fe048df`) — SM100 kernel plumbing: `kvcache_mode` compile-time
  flag, three optional runtime tensors (`mK_new`, `mV_new`, `mCache_seqlens`)
  threaded through `FlashAttentionForwardSm100.__init__/__call__/kernel`.
  `use_tma_KV=False` auto-flip when `kvcache_mode=True`. Loader body NOT
  changed; `const_expr(self.kvcache_mode)` gate keeps the False path
  byte-identical to upstream.

Not landed: loader-body semantics (Task 3b), SM120 mirror (Task 4),
interface wiring (Task 5), edge tests (Task 6), benchmark (Task 7),
README/gate log (Task 8).

## Why paused

FA4 has no cp.async smem→gmem primitive — the only async smem→gmem path
in the CuTe DSL is TMA bulk store (`CopyBulkTensorTileS2GOp`), which
lacks per-row predication. Every viable write-back design (TMA bulk +
rmem-fallback boundary, or full rmem-intermediate with pipeline_kv API
surgery) is more invasive than the design spec anticipated.

Meanwhile, the JoyBrain-side estimation that ruled out application-layer
SDPA was wrong: it assumed per-step `torch.cat` of full KV. The correct
SDPA pattern is `k_cache[:, cache_len:cache_len+seq_new] = k_new` +
view-slice into SDPA — a few KB memcpy per attention call, well within
HRM's 300 ms/token SLO on SM12.

HRM inference is now going the SDPA route (JoyBrain repo,
`veomni/models/transformers/hrm/core/layers.py`,
`HRM_ATTN_BACKEND=sdpa`).

## Resuming this work

If SDPA misses SLO on real profiling, this branch is the resume point:

1. `git checkout fa4-kvcache-non-paged` at `fe048df` — plumbing is done.
2. Read the recon report (in the JoyBrain repo,
   `.superpowers/sdd/task-3-recon.md`) — it has all the file:line
   anchors and design tradeoffs already worked out.
3. Task 3b re-scope proposal is in `.superpowers/sdd/task-3b-report.md`
   (also JoyBrain repo). Choose write-back primitive: TMA-bulk S2G with
   rmem-fallback for boundary tiles (recommended) vs full rmem
   intermediate (needs pipeline_kv surgery).
4. Proceed with plan Tasks 3b → 4 → 5 → 6 → 7 → 8.

## Install (as of pause point)

```
pip install git+https://github.com/KaijingOfficial/flash-attention@fa4-kvcache-non-paged
```

Note: `flash_attn.cute.flash_attn_with_kvcache_func` will import cleanly
and validate its inputs, but the kernel wiring is not complete — calling
it will raise `NotImplementedError("kernel wiring pending; will land in
Task 5 of the fa4-kvcache-non-paged plan.")`. It is NOT ready for
production use.
