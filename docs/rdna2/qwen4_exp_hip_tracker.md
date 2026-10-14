# Qwen4Exp HIP path — work tracker

**Scope**: track what needs to be tested and optimized for the HIP
scaffolding landed 2026-09-14 (`hc_rdna2.cu`, `qsa_rdna2.cu`,
`ple_short_conv_rdna2.cu` + bindings + dispatchers + env-var gates).

The matching test specification lives at
[`qwen4_exp_hip_tests.md`](qwen4_exp_hip_tests.md); the matching
path overview lives at
[`qwen4_exp_hip_path.md`](qwen4_exp_hip_path.md). Update this tracker
whenever a kernel moves between stages.

## Status legend

- **[ ]** open
- **[~]** in progress (someone is looking at it now)
- **[x]** done
- **[!]** blocked / wrong — needs investigation before continuing

## Stage gates

| Stage | Description | Exit criteria |
|---|---|---|
| **S0 build** | Code compiles, ops registered | `test_rdna_hip_build_smoke` passes |
| **S1 parity** | HIP output matches Triton reference (fp16 noise) | Parity test passes for each kernel |
| **S2 dispatch** | Env-var gate works, dispatcher routes correctly | `test_*_env_gate` tests pass |
| **S3 integration** | Real forward pass completes without raising | `test_qwen4_exp_rdna_hip_smoke` passes |
| **S4 cudagraph** | Captured graph replays without fault | `test_vllm_serve_smoke` passes |
| **S5 bench** | HIP throughput ≥ Triton (or justified slower) | Bench numbers logged in `qwen4_exp_hip_path.md` |
| **S6 default-on** | Flip env-var default to "1" + remove `VLLM_DISABLED_KERNELS` parallel | All tests pass with default-on |

## S3 serving A/B — the gates measured end to end (2026-10-13)

`tools/rdna2/bench_serve_gates_ab.sh`, production server
(`tools/rdna2/serve_v620_stable.sh`, TP4/EP4 + MTP2, Uncensored W4A16, KV=4026531840),
one fresh server process per arm (both gates are read at import time), same
box and session, `vllm bench serve` with `--dataset-name random --seed 0
--max-concurrency 1 --ignore-eos`. Four arms:

| arm | `VLLM_RDNA_QSA_HIP` | `VLLM_GDN_HIP_PREFILL` |
|---|---|---|
| A | 0 | 0 |
| C (A repeat) | 0 | 0 |
| D | 0 | 1 |
| B | 1 | 1 |

| metric | A | C | D (GDN) | B (both) | D vs off | B vs off |
|---|---:|---:|---:|---:|---:|---:|
| prefill 8k, total tok/s | 1737.5 | 1709.1 | 1630.8 | 1624.0 | **-5.4 %** | -5.8 % |
| prefill 16k, total tok/s | 1809.6 | 1801.2 | 1697.7 | 1705.7 | **-6.0 %** | -5.5 % |
| prefill 16k, TTFT p50 ms | 9555.2 | 9624.3 | 10125.7 | 10139.6 | **+5.6 %** | +5.7 % |
| decode 128+256, TPOT p50 ms | 11.83 | 11.81 | 11.60 | 10.67 | -1.9 % | **-9.7 %** |

The two off-arms agree to 0.5-1.7 % on prefill and 0.2 % on TPOT, so the
differences above are real at this workload.

**What this says**

1. **`VLLM_GDN_HIP_PREFILL=1` costs ~5.5 % prefill and buys nothing.** D and B
   are indistinguishable on prefill, so the whole regression is GDN, and QSA
   contributes nothing measurable there. Decode is unaffected by D, as
   expected (no GDN prefill in a decode step).
2. **Almost none of that 5.5 % is `delta_h`.** The delta_h kernel is +1.0 ms
   per layer at 8k prefill (4262 vs 5269 us) over 36 GDN layers = ~36 ms of a
   ~9.6 s prefill, i.e. ~0.4 %. The other ~5 % is in
   `gdn_prefill_{prep,kkt,solve_wy,o}_rdna2`, which nobody has profiled yet.
   **That is the next thing to look at**, and it is the reason the draft PR's
   "-8.7 % on a 16k prefill" survives despite delta_h being fixed.
3. **`VLLM_RDNA_QSA_HIP=1` buys ~9.7 % decode TPOT for free.** D vs B on decode
   isolates it (11.60 vs 10.67 ms); prefill is unchanged within noise. At the
   row counts a decode step actually uses, this is *not* the `mqa_paged` win
   measured at rows=512 — those shapes do not occur in decode — so the cause is
   still unidentified and is worth its own profile.

Caveat: the repo's own notes record ~10 % session-to-session drift on these
serving numbers, so only the within-session A/B above is meaningful, and it is
one workload (concurrency 1, no prefix-cache hits).

---

## HC prefill HIP

Code: `csrc/rocm/hc_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/hc_rdna2.py` · Gate:
`VLLM_RDNA_HC_PREFILL_HIP=1`

### hc_grouped_gemma_rmsnorm_rdna2

- [ ] **S0 build** — rebuild `_rocm_C.abi3.so`, run op-presence probe
- [ ] **S1 parity** — write
  `tests/kernels/rocm/qwen4_exp/test_hc_rdna2.py::test_grouped_gemma_rmsnorm_parity`
  against `torch.ops.vllm.qwen4_exp_grouped_gemma_rmsnorm`. Cover
  `N ∈ {1, 8, 64, 1024, 8192}`, `DIM ∈ {512, 4096, 8192}`,
  `W_SHARED ∈ {True, False}`.
- [ ] **S2 dispatch** — `test_hc_dispatcher_env_gate` (§1.4 of test plan).
- [ ] **S3 integration** — runs in the Qwen4Exp PLE forward path with
  `VLLM_RDNA_HC_PREFILL_HIP=1`. Specifically check the pre-mix path in
  `hyperconnection.py:mix()` (the `_rdna_fused_ok(xn)` branch is
  currently gated on M <= 8 — verify prefill (M > 8) routes to the new
  Triton-or-HIP path).
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, NG=4)`. Target
  ≥ +20% over Triton.
- [ ] **S6 default-on** — once S1-S5 pass, flip env-var default.

### hc_silu_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — straightforward elementwise; `atol=1e-3` should
  hold.
- [ ] **S5 bench** — production shape `(B=2048, DIM=320)`. Silu + 1/HC
  scale is bandwidth-bound; expect +20-40% over Triton (vec8 loads,
  no Triton block overhead).
- [ ] **S6 default-on**

### hc_gate_mix_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — reduction over HC streams; `atol=5e-3`. Cover
  `HC ∈ {1, 2, 4, 8}` (template-specialized). Add `test_hc_unroll_factor`
  for `hc_count=3, 5` → expect `TORCH_CHECK` "hc_count in {1,2,4,8} only".
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, HC=4)`. The
  constexpr unroll over `HC` should win over Triton's `static_range`;
  target +20%.
- [ ] **S6 default-on**

### hc_combine_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — elementwise affine; `atol=1e-3`.
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, HC=4)`.
- [ ] **S6 default-on**

### hc_combine_norm_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — round-trip boundary test (§1.3 of test plan):
  verify `hc_combine + grouped_gemma_rmsnorm == hc_combine_norm(out, y)`
  elementwise.
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, HC=4)`.
  Single-pass over the data should win over two launches; target
  +30-40% over the `(hc_combine, rmsnorm)` Triton sequence.
- [ ] **S6 default-on**

---

## QSA decode HIP

Code: `csrc/rocm/qsa_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/qsa_rdna2.py` · Gate:
`VLLM_RDNA_QSA_HIP=1`

**How the S5 numbers below are measured** — `tools/rdna2/bench_qsa_rdna2.py`,
GPU 1, 50 iterations per arm, both arms through the same production
dispatcher. It reports *two* numbers, and they disagree in sign, so be
explicit about which one is meant:

- **`kernel_us`** — summed GPU kernel time (`torch.profiler`). This is what a
  **cudagraph-captured** decode step pays, and the only fair kernel-to-kernel
  comparison. Use it for "is the HIP kernel faster".
- **`call_us`** — CUDA events around a batched loop. It includes ~20 us of
  Python + Triton launch overhead, which the HIP ops do not pay. That is what
  a **non-cudagraph** step pays, and it flatters HIP by a wide margin.

An earlier revision of this tracker quoted only `call_us` and therefore
claimed +35.6 % / +62.9 % where the HIP kernels are actually *slower* than
Triton. `/tmp`-style per-iteration `perf_counter() + synchronize()` loops are
the same trap in reverse: they add an arm-dependent constant. Do not trust a
HIP-vs-Triton delta that does not name its timer.

### qsa_store_cache_rows_rdna2

- [x] **S0 build**
- [x] **S1 parity** — pure scatter (no math). Covered `num_rows ∈ {1, 8,
  64, 512}`, `WIDTH ∈ {32, 64, 128}`, `PAGE_SIZE ∈ {16, 256}`,
  `num_blocks ∈ {1, 4, 64}`, plus 2/4/8-byte element sizes and non-unit
  width strides (the transposed RoPE-position view). Bit-exact vs
  `_store_qsa_rows_kernel` in `test_qsa_rdna2.py`.
- [x] **S2 dispatch** — `test_qsa_dispatcher_env_gate` +
  `test_dispatched_compress_and_store_match_gate_off`.
- [x] **S3 integration** — runs in `Qwen4ExpQSAAttention`'s indexer write
  path; the `[rows, 1, width]` squeeze lives in the dispatcher's compat
  wrapper.
- [x] **S5 bench** — production shape `(rows=512, WIDTH=128,
  PAGE_SIZE=256)` on GPU 1 (`tools/rdna2/bench_qsa_rdna2.py`): kernel
  **6.0 us Triton vs 6.8 us HIP (-10.6 %)**; call-level 28.5 us vs
  12.3 us (+57 %), i.e. entirely Triton launch overhead. So the kernel
  misses the `>= +30 %` target and is at parity, not ahead.
  This *was* 15.7 us (2.5x Triton): the kernel launched one row per
  256-thread block, so only the 16 threads holding a 16-byte vector were
  ever active. It now spans `256 / (WIDTH / 8)` rows per block, which is
  what closed the gap.
- [ ] **S6 default-on**

### qsa_compress_groups_rdna2

- [x] **S0 build**
- [x] **S1 parity** — sum with raw/state switch + first-position tail.
  Covered `compress_ratio ∈ {1, 2, 4, 8}`,
  `LOAD_ROPE_POSITIONS ∈ {False, True}`, all-NULL `state_table`,
  `head_dim ∈ {32, 128}`, `rows ∈ {1, 8, 64}`, `num_requests ∈ {1, 4}`.
  Bit-exact vs `_compress_qsa_groups_kernel` (maxdiff 0.0) in
  `test_qsa_rdna2.py`.
- [x] **S2 dispatch** — `test_dispatched_compress_and_store_match_gate_off`.
  The gate also requires fp16 (the kernel is `half*`-only), so bf16 keeps
  Triton.
- [x] **S3 integration** — runs in the indexer's `_update_and_compress`.
- [x] **S5 bench** — production shape `(rows=512, head_dim=128, ratio=4)` on
  GPU 1: kernel **13.6 us Triton vs 17.1 us HIP (-25.6 %)**; call-level
  73.2 us vs 29.8 us (+59 %), again Triton launch overhead. The kernel
  therefore misses the `>= +15 %` target. It is nonetheless 7-23x faster
  than the version this work started from, which used a per-thread
  `float acc[128]` array **and had every thread in the block recompute the
  whole row**; now one thread owns one output dim with a scalar
  accumulator.
- [x] **S5b correctness guard** — `head_dim > 128` used to be silently
  truncated to `BLOCK_D`; it is now a `TORCH_CHECK`.
- [ ] **S6 default-on**

### qsa_mqa_paged_rdna2

- [x] **S0 build**
- [x] **S1 parity** — the old wrapper delegated to
  `paged_mqa_logits_decode_rdna2` (DeepSeek V4): fp8-only, `H == 64 &&
  D == 128` only, per-head weights, row-indexed `block_tables`,
  `context_lens` visibility, no `visible_blocks`. That was a semantic
  mismatch, not a naming one — it raised
  `q_fp8 must be FP8 e4m3fn or uint8 (got c10::Half)` on the first call.
  Replaced by a dedicated kernel with the exact Triton contract
  `qsa_mqa_paged_rdna2(q, k_cache, page_table, token_to_req,
  query_positions, sequence_lengths, compress_ratio, num_columns,
  score_scale) -> (logits, visible_blocks)`. Parity is covered by
  `tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py::test_mqa_paged_parity`
  (model geometry 4 heads x 128, plus 1/2-head cases) and the
  cudagraph capture/replay test in the same file.
- [x] **S2 dispatch** — `qsa_mqa_paged` in
  `vllm/models/qwen4_exp/amd/ops/qsa.py` now routes to the HIP op behind
  `qsa_use_rdna2(q.dtype)`; bf16 keeps Triton because the HIP kernels are
  fp16-only.
- [x] **S3 integration** — wired; `test_dispatched_indexer_selection_matches_gate_off`
  drives the full `qsa_select_paged_tokens` (score -> top-k -> expand) both
  ways and the selected token sets agree. A real TP4 forward pass with
  `VLLM_RDNA_QSA_HIP=1` now also runs (see the S3 serving A/B section):
  decode TPOT p50 11.82 -> 10.67 ms, prefill unchanged within noise.
- [x] **S5 bench** — fp16, `H` free, `D <= 128` (indexer is 4 x 128),
  `compress_ratio = 4`, `rows=512`, 8192 columns, GPU 1: kernel **738.7 us
  Triton vs 73.5 us HIP (+90.0 %)**; call-level 746 us vs 231 us (+69 %).
  This is the one QSA op where HIP wins outright, and it wins on kernel time,
  not launch overhead: the HIP kernel skips the columns outside each row's
  visible range, while the Triton kernel masks them but still computes them.
  The dense fast path uses `v_dot2` (2 MAC/inst) when the rows are contiguous
  with even head_dim, with a scalar fallback for strided views.
- [ ] **S6 default-on** — deferred until the real-forward S3 run lands.

---

## GDN prefill HIP

Code: `csrc/rocm/gdn_prefill_delta_h_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/gdn_prefill_rdna2.py` · Gate:
`VLLM_GDN_HIP_PREFILL=1`

### gdn_prefill_delta_h_rdna2

- [x] **S0 build**
- [x] **S1 parity** — 30 cases in
  `tests/kernels/quantization/test_gdn_prefill_delta_h_rdna2.py`
  (`H,Hg ∈ {(12,4),(4,4)}` x 6 `seq_lens` x initial-state on/off). The
  original suite was itself wrong and had to be fixed before it could say
  anything about the kernel: it fed `g` as a *global* cumsum (the contract
  is chunk-local, `chunk_local_cumsum_scalar_kernel` is launched per chunk)
  and used unnormalised N(0,1) k/w, so the inter-chunk operator `I - k^T w`
  was O(100), `|h|` grew ~30x per chunk, and the Triton reference returned
  NaN from 4 chunks on.
- [x] **S5 bench** — `tools/rdna2/bench_gdn_prefill_rdna2.py`, GPU 1, TP4
  production geometry (`linear_num_value_heads=48`,
  `linear_num_key_heads=16` -> `H=12, Hg=4`), median of 50, CUDA events:

  | seq_lens | Triton us | HIP us | delta |
  |---|---:|---:|---:|
  | `[64,192,128]` | 229.0 | 268.4 | +17.2 % |
  | `[4096]` | 1989.6 | 2429.9 | +22.1 % |
  | `[2048,2048]` | 1278.8 | 2495.2 | +95.1 % |
  | `[8192]` | 4262.0 | 5268.6 | +23.6 % |
  | `[8192]x4` | 12727.7 | 19550.9 | +53.6 % |

  So HIP is ~20 % behind at single-sequence prefill, not the 5.5 % the draft
  PR claims, and ~2x behind when several long sequences share the batch.
  **This is now close to the hardware wall, not a fixable bug.** Both kernels
  are V_DOT2-issue-bound: the algorithm needs 1.6e9 `v_dot2_f32_f16`
  (2 MAC/inst) and gfx1030 issues 1/cycle/SIMD, so ~3.9 ms at [8192] is the
  floor. Triton sits at ~4.26 ms (92 % of that), HIP at ~5.27 ms (74 %).
  Evidence: deleting the v-correction dot alone (halving fdot2) drops HIP to
  3001 us; and cutting ~17 % of the loop's instructions by widening the k/w
  loads to 128-bit changed nothing, so the kernel is not instruction-count
  bound.
- [x] **perf fix landed** — the loop used a live `float v_corr[GDN_BT]` (64
  registers) across a fully unrolled 32-iteration h-update loop. That needed
  256 VGPRs + 427 spilled + 1632 B/lane of scratch, and cost 2.3-3.3x vs
  Triton at prefill scale. Fusing v-correction, gating and h-update into one
  t-pair loop (v-corr only needs the pre-decay `h_fp16`, already in
  registers) plus `#pragma unroll 8` on that loop needs 185 VGPRs and 0
  spill.
- [x] **S3 integration** — real TP4 forward pass with
  `VLLM_GDN_HIP_PREFILL=1` runs clean, but it costs **~5.5 % prefill**
  (1705.7 vs 1805.4 tok/s at 16k) and buys nothing on decode. See the S3
  serving A/B section. delta_h itself accounts for only ~0.4 % of that, so
  the rest is in the other four GDN prefill kernels.
- [ ] **S6 default-on** — do not flip: the gate is a measured 5.5 % prefill
  regression and the remaining kernels have not been profiled.

---

## PLE dilated short-conv HIP

Code: `csrc/rocm/ple_short_conv_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/ple_conv_rdna2.py` · Gate:
`VLLM_RDNA_PLE_CONV_HIP=1`

### ple_short_conv_decode_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — see §3.2 of test plan. Cover `B ∈ {1, 8, 64}`,
  `D ∈ {2560, 5120}`, `state_len ∈ {2, 4, 8}`, `dilation ∈ {1, 2}`,
  `silu ∈ {True, False}`, `has_init mix`, `null_block slots`.
- [ ] **S2 dispatch** — `test_ple_conv_env_gate`.
- [ ] **S3 integration** — already wired into
  `Qwen4ExpPLELayer._short_conv_dilated_decode_batched`. Smoke
  pass through the PLE forward path.
- [ ] **S5 bench** — production shape `(B=8, D=5120, K=5, dilation=2)`.
  Target ≥ +50% vs `F.conv1d` (depthwise over D=5120 channels with
  vec8 inner-loop, vs PyTorch's generic dilated conv path).
- [ ] **S6 default-on** — flip when S5 holds.

### ple_short_conv_prefill_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — see §3.3 of test plan. Cover `num_prefills ∈
  {1, 4, 8}`, `max_len ∈ {128}` (capped at 128 in current HIP build),
  state_len / dilation / silu matrices.
- [ ] **S3 integration** — wired into
  `_short_conv_dilated_prefill_batched`. Watch the early-return
  guards (`if ... pass  # fall through to torch`) — they should NOT
  trigger in normal traffic.
- [ ] **S5 bench** — `F.conv1d` is GPU-overhead-bound at small
  `max_len`; expect a smaller win than decode (target +20-30%).
- [ ] **S6 default-on**

---

## Open questions / blockers

- **Q1 (hc_combine_norm)**: the HIP wrapper writes `out` (post-combine)
  AND `y` (post-norm) in one launch. The Triton reference has the same
  contract. Verify on a checkpoint-loaded model that
  `hc_combine_norm(...).out` equals `hc_combine(...)` and `.y` equals
  `grouped_gemma_rmsnorm(out, w, eps, hc_count)`. If the model's next
  step is the `attn_hc.mix(...)` GEMV (the fused decode path), check
  that the `.y` produced here matches what `attn_hc._rdna_fused_ok`
  expects.
- **Q2 (ple_conv state write-back)**: the current decode HIP path uses
  `state[i] = history[dilation + i]` (left-shift by `dilation`). The
  eager torch reference uses `next_state = history[..., -state_len:]`.
  These differ when `dilation < state_len`. Pick the eager torch
  semantics as the source of truth and update the kernel.
- **Q3 (qsa_mqa_paged wrapper)**: RESOLVED 2026-10-13. It was not only a
  signature gap — the delegate kernel
  (`paged_mqa_logits_decode_rdna2`) required fp8 e4m3 and `H == 64,
  D == 128`, used per-head weights, row-indexed `block_tables`,
  `context_lens` visibility and had no `visible_blocks`. A dedicated
  kernel now implements the Triton contract exactly; see the
  `qsa_mqa_paged_rdna2` section above. Residual, unrelated: the
  `qsa_compress_groups_kernel` `BLOCK_D=128` ceilings `head_dim` (now a
  `TORCH_CHECK`, not silent truncation).
- **Q4 (HC `W_SHARED` layout)**: the current HIP path infers `W_SHARED`
  from `weight.numel()`. If a future checkpoint stores weights with a
  reshape (`[HC, GROUP_DIM]` flattened to `[HC * GROUP_DIM]` in some
  shape), the inference breaks. Add a TODO when we see a second
  checkpoint.
- **Q5 (cudagraph capture)**: none of the new HIP kernels allocate
  scratch or use Triton JIT, so they should be cudagraph-safe
  out-of-the-box. Confirm by capturing/replaying
  `_short_conv_dilated_decode_batched` with the dispatcher on.

---

## Next actions (priority-ordered)

1. **[S5 bench, highest value]** Profile
   `gdn_prefill_{prep,kkt,solve_wy,o}_rdna2` the way delta_h was profiled.
   The gates-on prefill is ~5.5 % slower and delta_h explains only ~0.4 % of
   it, so ~5 % is sitting in those four kernels. Each has a parity test
   already (`tests/kernels/quantization/test_gdn_prefill_*_rdna2.py`), so a
   bench harness is cheap. A V_DOT2/instruction-count analysis like the one
   in the GDN section is the fastest way to tell "bug" from "wall".
2. **[S5 bench]** Find what makes `VLLM_RDNA_QSA_HIP=1` worth ~9.7 % decode
   TPOT. The rows=512 `mqa_paged` win does not occur at decode row counts, so
   the cause is something else in the QSA decode path — profile a decode step
   with the gate on/off rather than guessing from the op benches.
3. **[S5 bench]** `compress_groups` is 25.6 % slower than Triton at the kernel
   level and nobody has looked at why.
4. **[S3]** Re-run the serving A/B at higher concurrency and with prefix-cache
   hits; the numbers here are concurrency-1 and zero-cache.
5. **[S6 default-on]** `VLLM_RDNA_QSA_HIP` is the only gate with a measured
   win (decode). `VLLM_GDN_HIP_PREFILL` is a measured 5.5 % prefill loss —
   leave both at "0" until items 1 and 2 land.
6. **[env]** The `TUNABLEOP_PROFILE=rocm10-rocblas5.6` pin in the production
   launcher now hard-fails (loaded rocBLAS hashes to `d05939967b5f`). Either
   re-register the new build's profile or drop the pin; production currently
   only starts with `TUNABLEOP_PROFILE=`.

---

## Change log

- **2026-09-14** — scaffolding shipped:
  - `csrc/rocm/hc_rdna2.cu` (5 kernels)
  - `csrc/rocm/qsa_rdna2.cu` (3 ops)
  - `csrc/rocm/ple_short_conv_rdna2.cu` (2 kernels)
  - `vllm/models/qwen4_exp/amd/ops/{hc,qsa,ple_conv}_rdna2.py` dispatchers
  - `csrc/rocm/torch_bindings.cpp` (10 new ops under `#ifdef VLLM_ROCM_GFX1030`)
  - `csrc/rocm/ops.h` (10 host-wrapper declarations)
  - `CMakeLists.txt` (3 new files in the gfx1030 `EXT_SRC` list)
  - `vllm/envs.py` (3 env vars: `VLLM_RDNA_HC_PREFILL_HIP`,
    `VLLM_RDNA_QSA_HIP`, `VLLM_RDNA_PLE_CONV_HIP`; default "0")
  - `vllm/models/qwen4_exp/amd/ops/{hc,qsa}.py` (dispatcher early-returns)
  - `vllm/models/qwen4_exp/amd/ple_layer.py` (early-return branches in
    `_short_conv_dilated_decode_batched` /
    `_short_conv_dilated_prefill_batched`)
  - `docs/rdna2/qwen4_exp_hip_path.md` (overview + opt-in recipe)
  - `docs/rdna2/qwen4_exp_hip_tests.md` (test specification)

---

## 2026-09-14 — QSA/attention abort diagnostic (pre-HIP-path blocker)

The `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` checkpoint now **loads, shards,
EP-splits, registers all 4 PLE workers, allocates KV (313k tokens) and captures
its CUDA graphs** after the PLE/offload/cache fixes in `rdna_extras`
(`e7ce54178`, `469c456d6`, `7259f0f44`). It then dies with:

- Worker `SIGABRT` (exit -6), no Python traceback; `GPU core dump failed`.
- dmesg: `amdgpu ... Trap debug id already reserved` (×3) — a GPU **runtime**
  trap, **not** a Triton compile abort (no `Fatal Python error`/`make_amdgcn`).

Observations:
- MoE already on RDNA2 HIP (`CompressedTensorsWNA16RDNA2MoEMethod`).
- `Using FlashAttention version None` (no flash-attn lib on ROCm).
- QSA forward routes via `qwen4_exp_qsa_with_output` (registered as a
  `direct_register_custom_op`) -> Triton `qsa_sparse_paged_attention`
  (`forward_qsa` in `vllm/models/qwen4_exp/amd/qsa.py`).
- `Op 'sparse_attn_indexer' doesn't exist` is a **harmless no-op** (that's the
  CDNA/AITER config name; the AMD QSA uses `paged_mqa_logits_decode_rdna2`).
- **QSA tile env overrides did NOT fix it**: `VLLM_RDNA_QSA_BLOCK_N=16
  VLLM_RDNA_QSA_WARPS=2 VLLM_RDNA_QSA_SPLITS=1` still SIGABRTs. So it is not
  the prefill tile profile.

This is the `[!]`-status item to address when finalising the HIP path: the QSA
`forward_qsa` / `qwen4_exp_qsa_with_output` runtime (the `qsa_rdna2.cu` ops
under `VLLM_RDNA_QSA_HIP` are the target replacement).

**Standalone-probe refinement (2026-09-14):** `qsa_sparse_paged_attention`
(the `forward_qsa` sparse-attention kernel) **passes in isolation** with the
model's actual geometry (24 heads / 2 KV / head_dim 256 / group 12, up to 512
tokens) — no trap. So the SIGABRT is **not** the sparse-attention kernel; it is
upstream of it in `_run_qsa`: either the **MQA paged indexer**
(`self.indexer(...)` -> `_qsa_mqa_paged_kernel`) or the `do_kv_cache_update` /
FlashAttention-forward integration. Probe: `/tmp/qsa_probe.py` on `.176`.

**Refinement 2 (2026-09-14):** `qsa_mqa_paged` (the indexer, `BLOCK_N=128 /
BLOCK_D=256 / num_warps=8`) **also passes standalone** with the model config
(24 heads, head_dim 256, up to 64 tokens). So neither QSA kernel is the trap;
the SIGABRT is in the **forward integration** — the `qwen4_exp_qsa_with_output`
op -> `_run_qsa`'s `do_kv_cache_update` / FlashAttention `forward` path, or a
different kernel in the model's warmup forward (the trace shows
`Using FlashAttention version None`). Probes: `/tmp/{qsa,mqa}_probe.py`.

**Refinement 3 (2026-09-14):** `triton_reshape_and_cache_flash` (the
FlashAttention KV write used by `do_kv_cache_update`) **also passes
standalone** (24 heads, head_dim 256, up to 512 tokens). So the KV-write,
indexer and sparse-attention kernels are all exonerated. The SIGABRT is in the
model's **warmup forward with real metadata / captured-graph context** — either
the `qwen4_exp_qsa_with_output` op's `_run_qsa` integration, the FlashAttention
`forward` (`version None`), or another kernel in the warmup path. This is the
HIP-path target; a full `torch.profiler`/`rocprofv3` capture of the warmup
`warmup_kernels` step (with `VLLM_USE_BREAKABLE_CUDAGRAPH=1` to keep GDN/FA
eager) is the next diagnostic. Probes: `/tmp/{qsa,mqa,rvc}_probe.py`.

**Refinement 4 (2026-09-14):** `qsa_sparse_paged_attention` passes with the
**real `forward_qsa` cache layout** (`kv_cache [B,H,N,2D] ->
transpose(1,2).split(hd) -> canonicalize_singleton_dim_strides`), up to 512
tokens / 24 heads / head_dim 256. So the strided-view hypothesis is also out.
All QSA-adjacent kernels are exonerated; the SIGABRT is a different kernel in
the model's warmup forward (GDN / MoE / sampler / n-gram). Probe:
`/tmp/qsa_transpose_probe.py`.

---

## 2026-10-13 — QSA HIP: SIGABRT closed, `qsa_mqa_paged` contract fixed

**The 2026-09-14 SIGABRT is not live and was never a QSA-kernel fault.**
That entry is labelled "pre-HIP-path" for a reason: the abort happened on the
default **Triton** path. `qwen38_flash_next_triton_status.md` (2026-09-15)
identifies it as **cudagraph capture**, not the model —
"the earlier SIGABRT was the cudagraph capture (a capture-unsafe op in the
QSA forward), not the model — eagerly it runs correctly" — fixed by
`8dc656b61` (UVA staging host VAs recorded into the captured graph). The
production configuration is `cudagraph_mode=PIECEWISE` + prefix caching:
16/16 successful, 0/18 sequential correctness, 8/8 coherence.

Confirmed for the HIP path too: the three `qsa_rdna2` ops are capture- and
replay-safe under `torch.cuda.CUDAGraph` (no scratch allocation, current
stream), pinned by
`tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py::test_qsa_hip_ops_are_cudagraph_safe`.

Changes:
- `csrc/rocm/qsa_rdna2.cu`: new `qsa_mqa_paged_kernel` implementing the
  Triton `_qsa_mqa_paged_kernel` semantics (fp16, free `H`/`D`, request
  indirection through `token_to_req`, visibility from
  `min((pos+1)/ratio, seq_len/ratio)`, ReLU-then-sum over heads,
  `score /= score_divisor`, `-inf` masking, `visible_blocks` side channel).
  Replaces the delegation to `paged_mqa_logits_decode_rdna2`.
- `csrc/rocm/qsa_rdna2.cu`: `qsa_compress_groups_kernel` was writing
  `float acc[BLOCK_D]` (128 floats/thread -> local-memory spill) and **every
  thread in the block recomputed the same row**. Replaced with one thread per
  output dim and a scalar accumulator (bit-identical output, 7-23x faster).
- `qsa_compress_groups_rdna2` now `TORCH_CHECK`s `head_dim <= 128` instead of
  silently truncating to `BLOCK_D`.
- `qsa_use_rdna2(dtype)` gates on fp16; the HIP kernels are fp16-only, so a
  bf16 model now keeps Triton instead of hitting a `TORCH_CHECK`. All three
  `ops/qsa.py` call sites pass their tensor dtype.
- `tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py`: op-presence probe, gate
  tests, dispatcher routing, S1 parity for all three ops (bit-exact for
  store/compress, maxdiff <= 2e-6 for the fp32 logits), `-inf`/mask
  equality, `head_dim` ceiling, end-to-end indexer selection parity, and the
  cudagraph check. 193 tests.
- `csrc/rocm/ops.h`, `csrc/rocm/torch_bindings.cpp`, `vllm/_custom_ops.py`:
  new `qsa_mqa_paged_rdna2` schema returning `(logits, visible_blocks)`.

Measured on GPU 1 — **kernel time**, `tools/rdna2/bench_qsa_rdna2.py`
(fp16). The call-level numbers this entry originally quoted were ~20 us of
Triton launch overhead, not kernel time; see the methodology note in the QSA
section:

| op | shape | Triton | HIP | delta |
|---|---|---:|---:|---:|
| `store_cache_rows` | rows=512, w=128 | 6.0 us | 6.8 us | -10.6 % |
| `compress_groups` | rows=512, d=128, r=4 | 13.6 us | 17.1 us | -25.6 % |
| `mqa_paged` | rows=512, cols=8192 | 738.7 us | 73.5 us | **+90.0 %** |

Superseded on 2026-10-13: S3 is done (see the S3 serving A/B section) —
`VLLM_RDNA_QSA_HIP=1` is worth ~9.7 % decode TPOT, and
`VLLM_GDN_HIP_PREFILL=1` costs ~5.5 % prefill. Both defaults stay "0" until
the remaining GDN prefill kernels are profiled.
