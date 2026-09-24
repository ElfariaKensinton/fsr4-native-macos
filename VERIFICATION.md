# Verification against the DX12/WARP golden dumps

Comparison of `fsr4mac` output against the reference produced by the unmodified
AMD FSR4 DX12 provider running on the WARP software adapter
(`ref/testcases/…`, FidelityFX-SDK 01446e6, provider `4.0.2`, INT8 `v07-i8`
path). Reference dumps include per-frame outputs, per-pass scratch snapshots
(pass00=pre … pass13=post), recurrent/history/reprojected state and the
resolved `OptimizedConstants`.

Machine: VirtualMac2.1 (Apple M4 virtualized, "Apple Paravirtual device"),
Xcode 26.6.

## Deviations found and fixed

1. **float3 texture-store expansion (alpha channel)** — `RWTexture2D<float3>`
   stores (`rw_mlsr_output_color`, `rw_history_color`, `rw_reprojected_color`,
   `rw_rcas_output`) expand to 4 components with **alpha = .x** on the D3D12
   side; spirv-cross emitted `.xyzz` (alpha = blue). Fixed in
   `tools/compile_variant.py`: `.xyzz, uint2` → `.xyzx, uint2` on generated
   MSL. Verified against the golden: reference alpha == r at all 2,073,600
   pixels of `f00_out`/`f00_history`/`f00_reprojected`.

2. **Exposure input** — mirrored the reference harness: always bind an
   explicit 1×1 `R32_FLOAT` exposure texture with `fNN_exposure.bin` when
   present, else `meta.pre_exposure`. The provider's internal
   default-exposure path is not used (it produced garbage on WARP, and is now
   irrelevant on Metal too).

3. **Pass-dump naming** — `--dump-passes` now writes
   `fNN_pass00_scratch.bin` (pre) through `fNN_pass13_scratch.bin` (post) to
   line up 1:1 with the reference passdumps, plus `fNN_constants.bin` (raw
   `OptimizedConstants`) and `fNN_reprojected.bin`.

4. **Constants** — field-by-field identical to `fNN_constants.json`
   (inv_size, scale, inv_scale, jitter, mv_scale=[1,1], tex_size,
   max_renderSize, dims, reset, exposures, rcas flags).

## Measured results

`testcase_quality_720to1080` (quality, 1280×720 → 1920×1080, 8 frames),
worst-frame numbers (f07):

| channel | max-abs | mean-abs | PSNR |
|---------|---------|----------|------|
| R | 0.39 | 0.0019 | 44.0 dB |
| G | 0.20 | 0.0025 | 46.3 dB |
| B | 0.32 | 0.0038 | 44.4 dB |
| A | 0.39 | 0.0019 | 44.0 dB |

`testcase_perf_540to1080` (performance, 960×540 → 1920×1080), worst frame:

| channel | max-abs | mean-abs | PSNR |
|---------|---------|----------|------|
| R | 0.20 | 0.0026 | 42.7 dB |
| G | 0.26 | 0.0031 | 44.1 dB |
| B | 0.63 | 0.0047 | 41.6 dB |
| A | 0.20 | 0.0026 | 42.7 dB |

Per-pass bisection (f00, quality): `pre` is essentially bit-exact — the
reprojected texture differs in only **533 / 8,294,400** values, all ≤1 fp16
ulp (max 4.9e-4). Differences then amplify through the conv network:

| pass | differing bytes | ±1-LSB share |
|------|-----------------|--------------|
| 01 | 2.2 % | 81 % |
| 05 | 23.4 % | 78 % |
| 09 | 42.3 % | 69 % |
| 13 | 62.8 % | 65 % |

Delta distribution is symmetric (±1 equally common, then ±2, …) — the classic
signature of **int8 quantization-boundary flips** seeded by sub-ulp fp16
differences, not a structural/pipeline bug: no spatial bias (diffs uniform in
background, moving square and edges alike, ~5× enrichment only in a <8 px
screen border), no sign bias, constants identical.

## Follow-up: PNG conversion resolved

The earlier side-by-side PNGs showed a washed-out look on the `fsr4mac` half.
Root cause identified and fixed — **not a color-space or pipeline difference**:

- Reference PNGs are written with **alpha = 255** (opaque). My PNG path was
  writing the data alpha channel, which is `alpha = r` (the D3D12
  `RWTexture2D<float3>` store-expansion semantics). Viewers composite
  PNG alpha over their background → my half rendered washed out.
- RGB encoding paths are equivalent: decoding the reference `f07_out.png` and
  re-encoding the *same* reference `f07_out.bin` through my converter
  (`src/bin2png.cpp`, identical sRGB curve as `fsr4mac`) gives RGB mean 1.15/255,
  max 3/255 — within encoder rounding. The entire apparent difference was
  the alpha channel (alpha-only mean diff 68.8/255).
- Metric sanity checks: `compare` on ref-vs-ref = 0 exactly; per-channel means
  of `f07_out.bin` are identical to 4 decimals (R 0.5367, G 0.3380, B 0.4035,
  A 0.5367) — no global chroma shift in the data itself.
- `fsr4mac` and `bin2png` now write opaque-alpha PNGs. The new side-by-side
  (`sbs2_f00.png`, `sbs2_f07.png`) is composed of both `.bin` files run
  through the single `bin2png` code path.

## Follow-up: self-noise quantified

Two identical `fsr4mac --dump-passes` runs on `testcase_quality_720to1080`,
same machine, back to back:

| comparison | max-abs | mean-abs | PSNR |
|-----------|---------|----------|------|
| run A vs run B, f00_out | 0.099 | 4.5e-5 | 62.8 dB |
| run A vs run B, f07_out | 0.333 | 4.5e-4 | 52.3 dB |
| run A vs reference, f07_out | 0.39 | 1.9e-3 | 44.0 dB |

Per-pass bisection of run A vs run B (f00 scratch dumps): **bit-identical
through pass10; first divergence at pass11** (39,015 differing bytes of
scratch), larger at pass12 (143,865), stable at pass13 (143,865 — post is
deterministic given its input). Nondeterminism enters inside the conv pass 11
kernel on this paravirtual GPU.

Interpretation: self-noise (PSNR ~52 dB, mean 4.5e-4) is ~4× smaller than the
reference diff (PSNR ~44 dB, mean 1.9e-3), so the residual is **not fully
explained by VM nondeterminism** — it is a real numeric deviation, but with
the identical quantization-flip signature (±1 int8 LSB at weight/activation
round boundaries, ~80 % of differing bytes ±1 LSB, sign-symmetric, no spatial
bias). The VM noise floor contributes a minority share of it.

## mpp::tensor_ops FNB path (native MSL, Metal 4)

`src/fnb_mpp.metal` replaces the generated `f16_FNB<NF,NG>` passes with a
matmul2d kernel (scalar dw conv staged to threadgroup memory + matmul2d for
the two pw convs, weights pre-transposed to [K][N] by gen_f16.py's
`mpp_manifest.json`/`mpp_weights.bin` sidecar). Auto-enabled on the fp16 path
when the sidecar + metallib exist; `--no-mpp` forces the generated passes.
Covers plain-FNB passes (quality@1080: 4,5,7,8,10); FNB_CT2D_ADD passes
(9,11) still use the generated kernels.

Accuracy vs WARP golden (quality testcase, paravirtual VM): f00 47.4 dB,
f07 41.9 dB all-channel PSNR — same band as the generated fp16 path.

**Negative result (measured on M5, `d32e075`):** mpp FNB loses to the
generated scalar fp16 kernels — the pw convs are skinny GEMMs (K,N =
32/64/128 channels) that don't fill the tensor-op tiles:
- tile=32px/threadgroup: mpp 2.28/2.35/2.46/3.92/2.37 ms (sum 13.4) vs
  generated 1.14/1.18/1.02/1.09/1.20 (sum 5.6); frame 28.9 vs 21.2
- tile=128px (NF32) / 64px (NF64): mpp 2.82/2.97/2.87/3.08/2.85
  (sum 14.6); frame 30.4 vs 21.3
The microbench advantage (2099 vs 281 GFLOP/s) does not transfer to
conv-shaped workloads — channel dims are too small and dw-conv staging
dominates. Default stays `--mpp` opt-in (generated path is default).

Note: matmul2d produces exactly ONE 32x32 output tile per `run()` call
(descriptor tile) — M>32 requires looping 32-wide slices in-kernel.

## Physical-hardware verification (Apple M5)

Same commit (`main`) run by the user on a physical **Apple M5, macOS 27.0,
Xcode 27.0**:

- **Bit-stable across identical runs** — all 8 `fNN_out.bin` identical
  between two runs. The pass-11 nondeterminism measured above is confirmed
  as a **paravirtual-VM artifact**, not a pipeline race.
- **Per-channel means of `f07_out.bin` match the WARP reference to 4
  digits** (R 0.5367, G 0.3380, B 0.4035).
- `MTL_SHADER_VALIDATION=1` does **not** zero output on real hardware —
  the all-zeros behaviour is a paravirtual-driver bug. However the
  instrumented run still deviates from the clean run (f07 PSNR 46.6 dB), so
  shader validation **perturbs results even on real HW** — never diff
  golden dumps against a `MTL_SHADER_VALIDATION=1` run.
- **~120 ms/frame at 720→1080 on the M5** — the earlier ~100–120 ms VM
  figure was therefore real hardware cost, not a VM artifact. This is the
  baseline for the optimization phase. (GPU timestamps on the paravirtual
  device itself are unreliable once shader caches are warm — they have been
  observed to under-report by orders of magnitude.)

## Known residual causes

- **fp16 arithmetic rounding**: Metal `half` ALUs vs WARP's fp16 emulation
  differ at sub-ulp level in `pre`/`post` (exp/log/sqrt paths) and inside
  dequant/quant scale math in the conv passes; each flip is amplified to a
  full int8 step by quantization. Experimentally, forcing `precise::` exp/log/
  sqrt/powr and `-fno-fast-math` both made the match *worse* (mean-abs
  0.009 / PSNR 28 dB), so the current fast-math MSL is closer to WARP than
  strictly-precise code is.
- **Paravirtual-GPU nondeterminism**: quantified above — identical runs are
  bit-exact through pass 10 and diverge from pass 11 (self-noise on
  `f07_out`: max 0.33, mean 4.5e-4, PSNR 52 dB). A minority of the residual
  sits below this device's own reproducibility floor; the majority is a real
  fp16-path deviation with the same quantization-flip signature.
- **Uninitialized/NaN lanes**: ~92 % of pass00 scratch u32 diffs are
  NaN-vs-NaN (different NaN encodings) in dead lanes never read downstream.

## Verdict

Pipeline is functionally correct: identical dispatch order, constants,
resource layout and per-pass semantics; pre-pass output matches to 1 fp16
ulp at 0.006 % of positions; outputs track the reference at PSNR ≈ 42–50 dB
(mean-abs ≈ 0.002, a few fp16 ulp at ~0.5 luminance) with up-to-~0.5 outliers
at ~0.5–3 % of pixels. The residual is dominated by int8 quantization-boundary
flips on a different fp16 arithmetic path plus paravirtual-GPU noise — not a
logic defect. On physical Apple Silicon (M5) the pipeline is bit-stable and
the per-channel means match the WARP reference exactly; tightening the
remaining residual further would require bit-exact fp16 emulation of the
network math (i.e. rewriting AMD's generated shaders).

## Phase 2 — FP16 model path

New `--fp16` model path: weights offline-dequantized to fp16
(`tools/gen_f16.py` → `initializers_f16.bin`, every BufferStorage tensor's
int8 payloads repacked as `int8 * per-tensor fp32 scale` at 2× byte size, same
record layout) and hand-written real-domain operators in
`patches/hlsl/ml2code_runtime/fp16_ops.hlsli` replacing the int8_NHWC family.

### fp16 semantics decision

The fp16 path stores **real-domain** activations (no int8 counts in scratch),
fp32 accumulation everywhere, and re-applies the int8 quantization semantics
as real-domain requantization at every store the int8 ops quantized at:
`v' = clamp(round(v / s), [-128, 127]) * s` (relu variant clamps [0,127]).
This keeps fp16 values numerically locked to what the int8 pipeline holds —
a pure real-domain variant (requant dropped) was tried first and measurably
drifted more (f00 PSNR 34.5 dB vs golden vs 42 dB with requant), so requant
stays.

Scale plumbing: where the int8 code took the output scale from the output
tensor's `quantizationScale` field, the fp16 transform appends it as an extra
float argument (fp16 tensors have no scale field). Per-half split scales
(`quantFactor0/1`, `quantOutputScale0/1`) are preserved.

### fp16 vs int8 (same pipeline, per-pass scratch, frame 0)

fp16 activations compared against int8 counts × the pass's output scale(s):

| pass | op | max | mean |
|---|---|---|---|
| pre | encoder1 downscale conv | 0.015 | 6.6e-5 |
| 1 | ConvNextBlock | 0.116 | 8.5e-4 |
| 2 | ConvNextBlock | 0.280 | 4.2e-3 |
| 3 | FusedConv2D_k2s2b_QuantizedOutput | 0.108 | 4.5e-3 |
| 4 | FasterNetBlock | 0.292 | 0.010 |
| 5 | FasterNetBlock | 0.600 | 0.017 |
| 6 | FusedConv2D_k2s2b_QuantizedOutput | 1.10 | 0.022 |
| 7 | FasterNetBlock | 1.39 | 0.031 |
| 8 | FasterNetBlock | 1.38 | 0.040 |
| 9 | FNB_CT2D_ADD | 1.14 | 0.036 |
| 10 | FasterNetBlock | 1.26 | 0.043 |
| 11 | FNB_CT2D_ADD | 0.95 | 0.028 |
| 12 | ConvNextBlock | 1.34 | 0.038 |

Mean error tracks ~1 output-scale LSB per element at the late passes —
accumulating quantization-boundary flips, same signature as the int8 path's
own self-noise, amplified through the FNB chain.

### fp16 vs golden (final output, per frame)

`testcase_quality_720to1080` (vs WARP DX12 golden):

| frame | all-ch max | mean | PSNR |
|---|---|---|---|
| f00 | 0.306 | 0.0022 | 42.0 dB |
| f03 | 0.349 | 0.0028 | 42.7 dB |
| f07 | 0.406 | 0.0028 | 42.7 dB |

(remaining 5 frames in the same 41.9–43.3 dB band; full run in the bench dir)

`testcase_perf_540to1080`: f00 PSNR 40.1 dB (mean 0.0023), f07 PSNR 38.9 dB
(mean 0.0042, max 0.63).

For comparison the int8 path scores ~44–50 dB on the same golden — fp16 is
~2–4 dB below, entirely explainable by per-element quantization-boundary
flips (a ±1 LSB difference in a mid-pass activation is a ~scale-sized error
that convs then propagate).

### fp16 performance (VM, directional only)

`--bench 20`, `testcase_quality_720to1080`, VirtualMac paravirtual GPU
(second column is after the CNB-once post fix):

| stage | int8 ms | fp16 ms | fp16+postfix ms |
|---|---|---|---|
| fsr4_pre | 2.48 | 2.37 | 2.8 |
| pass1/2/12 (CNB) | 9.5 each | 3.7 each | 5.2 each |
| pass3/6 (fused conv) | 2.6/3.3 | 1.1/1.2 | 1.5/1.6 |
| pass4/5 (FNB<32,1>) | 6.8 each | 1.8 each | 2.4 each |
| pass7/8 (FNB<64,2>) | 8.9/9.1 | 2.1/2.2 | 2.9/3.0 |
| pass9/11 (FNB+CT2D+ADD) | 11.0/9.1 | 2.5/2.1 | 3.4/3.0 |
| pass10 (FNB<64,2>) | 6.8 | 1.8 | 2.5 |
| fsr4_post | 11.6 | 15.0 | 6.4 |
| **TOTAL** | **107.0** | **45.2** | **~47** |

(VM timing drifts between runs ±15 %; the post fix shows clearly in the
per-stage number: 15.0 -> 6.4 ms.)

fp16 is ~2.4× faster overall; `fsr4_post` (CNB+CT2D fused, 8 output channels
per dispatch thread) is now the largest single stage at 15 ms — it did not
speed up because its inner loop is scalar fp16 over 32-wide channel dots and
it also carries the sigmoid/mu-law/texture tail. On the VM both numbers are
directional; M5 numbers pending via `--bench` on real hardware.

### Known fp16-path caveats

- The int8-vs-fp16 residual is *not* zero — see the numbers above; the fp16
  path is the fast path, the int8 path remains the accuracy reference.
- `fsr4_post` was a 3rd of the frame because my first fp16 port recomputed
  the whole ConvNextBlock inside each of the 4 ConvTranspose taps; the i8
  original computes it once. Split into `f16_CNB` (once) + `f16_CT2D_tap`
  (per tap): post 15.0 -> 6.4 ms on the VM, accuracy unchanged
  (f00 42.3 dB, f07 42.7 dB vs golden).
- `mpp::tensor_ops`/`MTLTensor` (Metal 4 neural accelerators) and
  `simdgroup_matrix` paths are NOT implemented yet — fp16 is pure scalar
  ALU so far. `--caps` prints a `matmul2d` compile probe; `--bench-tensorops`
  dispatches and times a real GEMM on both paths (VM: matmul2d 2388 vs
  simdgroup 639 GFLOP/s — verified correct vs a CPU reference).

### M5 fp16 numbers (user's physical machine, commit 86baee7)

fp16 frame time 43.2 ms vs int8 120 ms (2.78×). Accuracy on the quality
golden: f07 R41.57/G43.75/B42.48 dB, f00 R 39.28 dB — a couple dB *below*
the same binary's numbers on the VM (41.9–43.3 dB). Likely cause: the
paravirtual GPU and the M5 take different fp16 fast-math paths, so each
lands a different distance from WARP's fp16 emulation; the residual is the
same quantization-flip pattern either way.

## 2026-09-23 — i8mpp path re-verified on the M5 after the post CT2D fusion

Machine: physical Apple M5, macOS 27.0, Xcode 27.0. Golden: release
`golden-v0.1` (`testcase_quality_720to1080`, `testcase_perf_540to1080`,
8 frames each, per-pass dumps for frames 0–1). Pipeline: `--i8mpp` (native
int8 tensor-ops model, pass 13 = post's CNB + CT2D, generated int8 pre/post).

Host vs `ffx_provider_fsr4_dx12.cpp` (re-read side by side): constant
buffer layout and every field, dispatch order and group counts, preset
thresholds (1.50/1.69/1.99/2.99 on the X ratio), `mv_scale =
motionVectorScale / mvSize`, jitter cancellation, `previous_preExposure`,
RCAS constants, SPD setup and the six pass-0 weight tables
(`src/pass0_weights.h` == the provider's `weights_*[256]`, checked by
script) all match. Two provider details were missing and are now
implemented in both hosts: `AlignUp(upscaleSize, 8)` for internal
textures/constants/dispatches, and `reset` forced on the dispatch that
creates the model (first frame; live path also on preset/flag change).
Neither changes the 1920x1080 test cases.

Final output vs golden (all channels, `compare`):

| frame | quality PSNR | perf PSNR |
|---|---|---|
| f00 | 47.05 dB | 48.65 dB |
| f01 | 45.47 | 43.50 |
| f02 | 44.75 | 42.72 |
| f03 | 43.99 | 42.73 |
| f04 | 44.17 | 43.19 |
| f05 | 44.07 | 43.63 |
| f06 | 42.95 | 43.41 |
| f07 | 41.93 | 41.85 |

(mean-abs 0.0012–0.0034; same band as the generated int8 path.) Two
identical runs are bit-identical on every frame.

Model passes on the golden per-pass dumps (`tools/bench/i8_test`, int8
bytes of the pass output region):

| pass | quality f00 | quality f01 | perf f00 |
|---|---|---|---|
| 1, 2, 3, 5, 6, 8, 9, 12 | 0 | 0 | 0 (9: 21 flips) |
| 4 | 0 | 1 flip | 0 |
| 7 | 0 | 1 flip | 0 |
| 10 | 1 flip | 3 flips | 4 flips |
| 11 | 367,640 (band) | — | 262,782 (band) |

All flips are ±1 LSB. Pass 11: 367,630 of the 367,640 differing bytes lie
in output columns x < 64 (the reference's FNB_CT2D_ADD<32,1> band, see
STATE.md); the other 10 are ±1 flips.

Pass 13 fusion: with the previous scalar requant kernels the fused pipeline
was bit-identical to the pre-fusion build on frames 0–3 and 5 (frame 4:
one output pixel differs by 2 unorm steps in recurrent because the
vectorised requant flips one CNB int8 value in 8.3M; frames 6–7 inherit it
through history: 72/69 dB self-diff). The model-parameter half4s are exact
(final colour identical while recurrent differs). Compiler math-mode flags
(`-fmetal-math-mode`, `-ffp-contract`, `-O2`) do not change the count;
`-ffp-contract=off`/`safe` make it 15,868, so the generated post contracts
FMAs the same way the native kernel does.

Bit-stability of the generated stages: the `[loop]` build (default) is
bit-identical to the previous build; `--keep-unroll 1` (post ~9% faster)
drifts at the fp16-ulp level (50 dB self-diff) with the same distance to
the golden.

## Lane-parallel CNB (`cnb_mpp`)

CNB16 (passes 1/2/12 at 960x540x16) re-implemented in `src/fnb_mpp.metal`
as a lane-parallel kernel: one simdgroup per output pixel, channels split
across lanes (K-split 8+8 for conv1, out-per-lane for the 16->32 expand,
K-split 16+16 for the 32->16 contract), `simd_shuffle_xor` reductions,
identical requant semantics (`f16_requant`/`f16_requant_relu`). Weights
are used in the original packed layout indexed by manifest strides.

Correctness vs DX12 golden (VM, default --fp16 run with cnb_mpp active):
quality f00 42.54 dB (all), f07 42.79 dB — same band as the generated
fp16 path (42.3-42.8). Perf on the VM is not meaningful (subgroup ops
emulated); M5 decides.
