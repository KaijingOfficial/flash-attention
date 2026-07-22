# fa4-kvcache-non-paged

FA4 (`flash_attn.cute`) drop-in of FA3's `flash_attn_with_kvcache`, targeting
HRM inference on SM12 dev + SM100 prod.

- Base: `Dao-AILab/flash-attention:main` @ `2409214a03797b168f648ea30df1adbc09ce658a`
- Design doc: JoyBrain repo `docs/superpowers/specs/2026-07-22-fa4-flash-attn-with-kvcache-design.md`
- Scope: contiguous (non-paged) KV, `causal=False`, MHA, inference-only,
  `num_splits=1`, 9-kwarg API subset.
- Not for upstream. Fork-only.

## Install
```
pip install git+https://github.com/KaijingOfficial/flash-attention@fa4-kvcache-non-paged
```
