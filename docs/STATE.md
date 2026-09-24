# FSR4-Metal — project state & handoff

Last updated: 2026-09-24 (M1–M5 selection and FP16 boundary fixes). Read this before continuing
optimization work — it contains everything a fresh session needs.

## 2026-09-24: automatic backend and portable FP16

CLI/live now share `fsr4_backend.h` + `fsr4_device.h`: auto chooses FP16
for Apple families 7–9 (M1–M4 and their larger variants), native INT8 for
family 10+ with Metal 4 and sufficient resources. Each pipeline is checked
against its real threadgroup limits. Native initialization failure falls
back to FP16; live retries the same encode and latches failure per device,
even for `FSR4_MODEL=i8mpp`. `--int8` remains the explicit generated
reference mode. `--backend-info` provides JSON; `--caps` now actually runs
and verifies an INT8 matrix product instead of compiling an empty probe.

Fixed a **separate FP16 over-dispatch race**: FNB stores past logical row
width wrapped into the following row (480/240-wide levels at 1080).
The new bounds checks make repeated FP16 runs deterministic. Post skips
off-frame blocks before reading the model tensor. This is unrelated to the
reference INT8 pass-11 band described below. Measured FP16 gain is only
0.7–1.4% on M5; the larger attempted rewrites did not improve performance
and were discarded. Details: [BENCHMARK.md](../BENCHMARK.md).

`make test` and `make test-gpu` cover policy and GPU failure/repeatability.
The 55-frame FP16 suite and classic/Metal 4 live tests passed on M5/Rosetta.
M1–M4 still need physical performance/driver validation; policy tests are
not measurements on those devices. SPD auto-exposure still has the earlier
partial-tile nondeterminism, despite defining its initial exposure state.

Use `python3 tools/precompile_models.py` to prepare the selected backend
(`--model all` prepares both). The live encoder checks source hashes on
session creation; the installer creates a fresh reference with this Mac's
selected model. Rebuild the hook after updating host code. The committed
CLI targets macOS 26 instead of inheriting the build machine's macOS 27.
Full instructions and model overrides: [HARDWARE.md](HARDWARE.md).

## 2026-09-24 follow-up: parallel post filter

An additional **3–5%** reduction versus the first optimization below, with
55/55 tested frames byte-identical. Same-session quality/performance/balanced
fused means: **4.83 → 4.67 / 4.86 → 4.63 / 4.48 → 4.33 ms**. Details and
validation are in [BENCHMARK.md](../BENCHMARK.md#2026-09-24-follow-up-parallel-post-filter).

- `compile_variant.py` selects `patches/hlsl/post_filter.hlsl` when the INT8
  model parameters are supplied by native pass 13. It uses a 16x16 group
  with one output pixel per thread. The host's group count stays unchanged;
  the thread count comes from the generated bindings, including in Metal 4.
- The post filter checks its complete 3x3 cache footprint once per pixel.
  The fallback still uses the original colour transform for out-of-cache
  footprints or input/output ratios greater than 1. Keep the barrier before
  the output-size check so partial edge groups participate fully.
- `i8mpp.metal` gives fixed extents to tile/weight tensors and packs saturated
  int4 results through char4. Input convolution extents remain dynamic to
  preserve padding at the actual frame boundary.
- **Preserve `<post_common.hlsli>` in the new post entry point.** Quoted
  inclusion finds the raw header next to it before the shadow tree and
  bypasses the compiler's `[unroll]` → `[loop]` policy, changing numerics.
- Cooperative accumulators, a pre colour cache, wider pre loops and adaptive
  post-cache filling were measured and discarded; see the benchmark log.

## 2026-09-24: cache post-filter colours and combine CT2D taps

The native INT8 path now caches `ApplyMuLaw(input * exposure)` in float32
threadgroup memory for the post filter. One 8x8 group covers 16x16 output
pixels; a 20x20 input cache replaces their repeated colour transforms.
Samples outside the cache, including input/output ratios above 1, use the
original expression. Keep the cache origin's conservative `floor` expression:
sharing the filter's half-pixel position expression changed FMA contraction
and caused a few half-ULP differences, which spread through temporal history.
All threads join the cache barrier before out-of-range output pixels are
skipped; edge groups no longer read missing model parameters.

Passes 9, 11 and 13 now multiply all four CT2D taps together. Weights are
`[input channel][tap][output channel]`; the wider GEMM reuses the existing
FNB/CNB accumulator storage. Pass 13 also removes the former 8-to-16 channel
padding. The `*_tap4` entry points deliberately differ from the old names
so stale weight manifests cannot be silently consumed. CLI/live manifest
mtime checks regenerate weights; run `tools/precompile_i8mpp.sh` after
updating to refresh the live path's cached pre/post variants, then restart
the game. No hook ABI or model weights changed.

Same-session M5 A/B measurements and validation are in
[BENCHMARK.md](../BENCHMARK.md#2026-09-24-post-filter-cache-and-packed-ct2d-taps).
The original 32x4 CNB tile remains: 32x2, 16x4, 16x8, 64x2 and 8x8 did not
give a consistent improvement. Low Power Mode was off for these measurements;
older 10+ ms measurements in Low Power Mode are not the baseline.

## 2026-09-23: post CT2D on the neural accelerators, requant vectorised

Measured on the M5 (`--i8mpp --bench 10 --reps 10`, same session; "before"
is the previous commit under the same protocol, quality case):

| stage | before | after (quality) | after (perf 540->1080) | after (balanced 1000x648->1712x1112) |
|---|---|---|---|---|
| fused frame mean / min | 5.95–6.3 ms | **5.57 / 4.97 ms** | 5.41 / 4.95 | 5.06 / 4.74 |
| per-stage sum | 6.2–6.35 | 5.20 | 5.17 | 4.77 |
| post (generated) | 1.11 | 0.86 | 0.86 | 0.78 |
| pass 13 (native) | 0.51 (CNB only) | 0.61 (CNB + CT2D + sigmoid) | 0.60 | 0.55 |
| pass 1/2/12 (CNB) | 0.50–0.55 | 0.37–0.41 | 0.38–0.40 | 0.34–0.36 |
| pass 4/5/10 (FNB32) | 0.28–0.31 | 0.22–0.24 | 0.21–0.23 | 0.21–0.22 |
| pass 9/11 (CT2D) | 0.43/0.49 | 0.34/0.42 | 0.35/0.42 | 0.30/0.39 |
| pre (generated) | 0.62–0.77 | 0.68 | 0.65 | 0.61 (unchanged) |

What changed:

- **Pass 13 is now `i8_cnb_ct2d`**: the post pass's ConvNextBlock *and* its
  2x2 s2 ConvTranspose (16 -> 4 model parameters + 4 recurrent channels) run
  in one native kernel; the CT2D taps are `matmul2d` [T][16]x[16][16] (N
  padded 8->16). The kernel writes the recurrent texture itself (texture slot
  0) and stores the four model parameters as half4 per output pixel at
  `post_ext_ct2d_off` = `GetScratchSize(tier)` — the scratch buffer is now
  `I8Mpp::scratch_bytes(tier)` long. The post variant is compiled with
  `-DFSR4_POST_EXT_CT2D` (key suffix `_pt`, replaces `_pc`) and only does
  `apply_model_filter` + history/output writes. The 128 emulated `dot4`s per
  output pixel that made the generated post ALU-bound are gone.
  Expressions are transcribed from the generated MSL (`half4(float4(acc)*s)
  + half4(bias)`, `float4(half4(1/(1+exp(-v))))`) and are exact for the
  parameters (final output bit-identical on frames 0–4 of the quality golden
  run).
- **Requant loops handle 4 channels per step** (int4 accumulator load, one
  packed uint store, bias loaded once per thread) in CNB, FNB and CT2D.
  Golden per-pass check unchanged (the golden dumps cover frames 0–1):
  f00 passes 1–9 and 12 bit-exact, 10 one LSB flip; f01 passes 1/2/3/5/6/8/9/12
  bit-exact, 4/7/10 one to three single-LSB flips; 11 the reference band. Against the *previous*
  scalar kernels the vector form flips one CNB int8 value in ~10^7 (frame 4
  of the quality run: 2 values), which then spreads through recurrent/history
  (self-diff vs the old build: f04 identical, f05 112 dB, f07 68.8 dB;
  distance to the DX12 golden unchanged, 47.1/44.0/41.9 dB on f00/f03/f07).
- `compile_variant.py --keep-unroll 1` keeps the source `[unroll]` hints for
  pre/post/rcas (model passes always get `[loop]`): post ~9% faster (1.11 ->
  1.02 ms) but DXC then contracts differently — output is not bit-identical
  to the `[loop]` build (50 dB self-diff, same distance to golden). Off by
  default; opt in if reproducibility against earlier runs does not matter.
- `fsr4mac --bench N --reps R` encodes every stage R times per command
  buffer (all stages are idempotent) — per-stage numbers were ±30% noisy
  from GPU clock ramping with one dispatch per buffer.
- Host now follows the provider on two details it skipped: internal
  textures/constants/dispatch sizes use `AlignUp(upscaleSize, 8)` (the app's
  output texture keeps its size; writes past its edge are dropped), and
  `reset` is forced on the dispatch that creates the model (first frame of a
  context/view and every preset or flag change of the live session). No
  effect on the 1920x1080 test cases (bit-identical).
- **Live path: RCAS on by default at FSR sharpness 0.4** (DLSS/MetalFX has
  no sharpness knob). `FSR4_SHARPNESS=<0..1>`, `0`/`FSR4_RCAS=0` disables.
  Same wiring as the CLI/provider: post writes mu-law colour to an RGBA16F
  intermediate, `fsr4_rcas` (always in the metallib) applies RCAS +
  RemoveMuLaw/exposure into the game's output; cbRCAS at +224 of the cb ring
  slot. `live_test` with sharpness 0.4 is bit-identical to the CLI run with
  `rcas:true, sharpness:0.4`; RCAS vs no RCAS on identical inputs: mean-abs
  0.0076, max 0.125 (40 dB). The installer runs `live_test` with
  `FSR4_SHARPNESS=0` because its reference frame has no RCAS.
- **`FSR4 Panel.app`** (`tools/crossover/panel_app.mm`, built by the
  installer into `dist/crossover/`): a Dock app with the hook's status and a
  live RCAS sharpness slider; the hook publishes `dist/crossover/status.conf`
  at 2 Hz and hot-reloads `panel.conf` by mtime (`src/fsr4_panel.mm`).
  `FSR4_PANEL=1` in the bottle makes the hook `open -g` the app once.
- Rejected this round: a second `hlsl_shadow` tree for unrolled pre/post
  as default (numerics, above). Tile-shape sweeps for the CNB conv were not
  done (32x4 stays).

Ideas left: pre (0.6–0.75 ms) is now the largest generated stage — its
downscale conv + reprojection could move to a native kernel the same way;
post's remaining cost is the 9-tap Gaussian resample with mu-law (27
`log`s per pixel); CNB tile shape (TW/TH template) untested.

## Mission (unchanged)

Standalone native Metal implementation of the FSR4 upscaler as a CLI tool
(default model path on M5: `--i8mpp`, see below):
`fsr4mac <testcase_dir> <out_dir>` consumes recorded frames (colour / depth /
motion vectors + params) and writes upscaled frames numerically matching the
reference DX12/WARP implementation. Correctness first, performance second.
CrossOver DLSS replacement (copied D3DMetal, bottle env, hook) is
[CROSSOVER.md](CROSSOVER.md).

Authority:
- **Correctness:** golden dumps in GitHub Release `golden-v0.1`
  (`testcase_quality_720to1080.zip`, `testcase_perf_540to1080.zip`,
  `testcase_README.md`). Per-frame `fNN_out.bin` + per-pass scratch dumps
  `fNN_passPP_scratch.bin` + `fNN_constants.bin`/`fNN_reprojected.bin`.
- **Performance:** the user's physical M5 (macOS 27.x) is the only perf
  authority. The dev VM is a VirtualMac2.1 paravirtual device — timings there
  are directional at best and *misleading* for subgroup/tensor ops.

Repo: `UzenUPoZiTiV4ik/fsr4-mac`, branch `main`, push directly.

## int8 model on the M5 neural accelerators (`--i8mpp`, 2026-09-23)

The 12 model passes plus the post pass's ConvNextBlock now run as native
Metal 4 kernels on `mpp::tensor_ops` (`src/i8mpp.metal`, host
`src/i8mpp_host.h`, weights/manifest from `tools/gen_i8mpp.py`). Same int8
NHWC scratch tensors, int32 accumulation and fp32 requant expressions as the
generated int8 HLSL passes (transcribed from their compiled MSL: every
literal scale product is folded in fp32 by the Metal compiler), so the
result is the reference int8 numerics, not the fp16 approximation.

Kernels: `i8_cnb` (passes 1/2/12), `i8_cnb_ct2d` (pass 13: post's CNB +
2x2 ConvTranspose + recurrent sigmoid), `i8_ds16`/`i8_ds32` (3/6, gathered
2x2 s2 conv as matmul2d), `i8_fnb32_5f`/`_3f`/`i8_fnb64_5f` (4/5/7/8/10),
`i8_ct2d64`/`i8_ct2d32` (9/11). One 128-thread threadgroup per tile;
`convolution2d` int8->int32 into threadgroup memory, `matmul2d` int8 for the
pointwise convs, requant in fp32 with `rint`. `convolution2d` facts learned
the hard way: `destination_dimensions` is the per-threadgroup tile,
`source_dimensions` only an upper bound (runtime tensor extents do the
bounds/zero padding), `set_offsets` = tile origin, destination sliced to the
tile; `groups==1` only (grouped convs use channel-sliced strided views);
stride-2 never matched a reference, hence the matmul2d gather for 3/6.
`newLibraryWithSource` served stale binaries across edits, so the library is
built with `xcrun metal` into `build/.../i8mpp_<hash>.metallib`.

Pass 13 = post's ConvNextBlock + ConvTranspose (see the 2026-09-23 section
above); the int8 post is compiled with `-DFSR4_POST_EXT_CT2D`
(`patches/hlsl/post_common.hlsli`, variant suffix `_pt`) and loads the model
parameters. Tensors are a fixed canvas per tier; the kernels tile only the
`upscale>>(level+1)` region (`I8Mpp::set_region`).

M5, same session, fused frame (i8mpp column: 2026-09-23 CT2D-fusion build,
mean / min over 80 frames):

| case | int8 (generated) | fp16 (generated) | i8mpp |
|---|---|---|---|
| quality 720->1080 | 47.5 ms | 12.7 ms | **5.6 / 5.0 ms** |
| balanced 1000x648->1712x1112 (Spider-Man) | — | 11.1 ms | **5.1 / 4.7 ms** |

Per pass (quality 1080, isolated): cnb 0.47, ds 0.05-0.07, fnb32 0.26,
fnb64 0.19, ct2d 0.36-0.46; model total ~3.5 ms; post 1.2 + post_cnb 0.6,
pre 0.8.

Accuracy vs the DX12/WARP golden (perf 540->1080, 8 frames): i8mpp
49.2/44.0/42.8/42.7/43.3/43.8/43.3/41.8 dB — equal to the generated int8
path (49.4/44.0/43.2/...), far above fp16 (39-41 dB). Per pass on golden
inputs: passes 1-8 bit-exact, 9/10/12 within 2-21 single-LSB flips, pass 11
differs only in output columns 0..63.

**Pass 11 band (reference bug, not ours).** The reference FNB_CT2D_ADD<32,1>
(4-scale overload) produces, for the first 32 threads of threadgroup 0
(input x<32, output x<64), values whose 2x2 outputs land one output row
higher and whose 3x3 conv term is missing (impulse-response probe with
`/tmp/refpass`); rows 0 match. The WARP golden shows the same band, so it is
in AMD's HLSL as compiled (probably an out-of-bounds local array), not in
the Metal port. `i8_ct2d32` computes the intended math there; it is the only
place the native path knowingly deviates from the reference.

Not done / ideas: pre's 2x2 conv and post's Gaussian resample are still
generated HLSL (pre 0.6–0.75 ms, post 0.8 ms after the CT2D move);
`I8_DEBUG=1|2|3` macro dumps a1/a2/c32 for a pass, `I8_DEBUG=4` makes pass
13 also store its CNB int8 output where the pre-fusion pass 13 put it
(`tools/bench/i8_test`, `FSR4_I8_DUMP`/`FSR4_I8_DUMP_TEX` dump the scratch /
recurrent texture, `FSR4_I8_MFLAGS` appends `xcrun metal` flags).
ANE/CoreML was rejected: CoreML does not use the Neural Engine under
Rosetta (the CrossOver game process is x86_64), and the per-frame
GPU->ANE->GPU round trip would cost more than the whole model now does.

Tools: `tools/bench/i8_test <gendir> <dumpdir> <frame> <pass> [iters] [dbg]`
(one pass on a reference dump: mismatch count + timing; build with
`clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation
tools/bench/i8_test.mm`, set `FSR4_ROOT`),
`tools/bench/mpp_bench2` (tensor-ops microbench), `tools/precompile_i8mpp.sh`
(the `_pt` variants the live path needs).

## Live path (CrossOver) — motion trails, 2026-09-22

D3DMetal encodes more than one MetalFX temporal scaler per game frame; one of
them carries a 4x4 RGBA8 placeholder instead of the motion field. The live
encoder now keeps history/recurrent per scaler object, skips placeholder
encodes (`FSR4_STUB=skip|metalfx|run|reset`), brackets its Metal 4 encoder with
queue-scope barriers, honours the descriptor flags (jittered MV, output-res MV,
auto-exposure via SPD) and logs a per-encode trace plus a GPU probe (exposure,
|mv| max, centre depth) into `FSR4_HOOK_LOG`. Constants are unchanged and
match the DX12 provider and OptiScaler's DLSS->FSR mapping (jitter and
motionVectorScale pass through; `mv_scale = scale / renderSize`).
`build/fsr4_*` caches are not content-keyed: `rm -rf` them after patch edits.

## Architecture

### Pipeline (per frame)

`pre` (pixels→enc0, Halton jitter, autoexposure out) → model passes 1–12
(scratch-buffer tensors, half fp16) → `post` (enc0 residual + upscaled
input → output; ConvTranspose taps + sigmoid/mu-law tail) → optional RCAS.
Auto-exposure adds an SPD chain before pre. Persistent state: `recurrent`
texture, `history`/`reprojected` history textures; constants in
`MLSR_Optimized_Constants` cb (byte-identical to DX12 dumps).

Enc0 960×540×16, enc1 480×270×32, enc2 240×135×64 (quality @1080).
Passes: 1/2/12 = ConvNextBlock16 (3×3 16→16, 1×1 16→32, 1×1 32→16+residual);
3/6 = fused strided downscale convs; 4/5/10 = FNB<32,1>; 7/8 = FNB<64,2>;
9/11 = FNB_CT2D_ADD (FNB + transposed-conv add). All model intermediates live
in one `ScratchBuffer` with byte-offset tensor views.

### Generated vs native kernels

Default fp16 path: **all** passes run the generated kernels produced by
`tools/gen_f16.py` (HLSL template → same DXC→SPIR-V→MSL route as int8).
Native MSL kernels in `src/fnb_mpp.metal` exist behind flags:

| flag | kernel | default | status |
|---|---|---|---|
| `--mpp` | `fnb32_mpp`/`fnb64_mpp` (matmul2d) | off | rejected on M5 (below) |
| `--cnb` | `cnb_mpp` (lane-parallel CNB16) | off | rejected on M5 (below) |

`bench_csv` stage names get `_mpp` suffix for any native-dispatched stage.
`gen_f16.py` emits `mpp_manifest.json` + `mpp_weights.bin` per variant
(pass tensor geometry, scales, weight blobs; FNB weights pre-transposed to
[K][N]; CNB weights in original strided layout).

### Flags

`--fp16`/`--int8` (default int8), `--mpp`, `--cnb`, `--sync
barrier|encoder|none`, `--bench N` (per-stage CB timing + fused-frame
`FUSED_FRAME_<sync>` lines, CSV `bench_csv,model,stage,gpu_ms`),
`--bench-tensorops` (timed matmul2d vs simdgroup_matrix GEMM, GFLOPS),
`--caps`, `--dump-passes`, `--builddir <dir>` (must point at
`build/shaders` — gen_f16 outputs live under it).

### Shader toolchain

DXC `-spirv` → binary patch `OpCopyLogical`→`OpCopyObject` → SPIRV-Cross
`--msl` → `xcrun metal`. Flags: `-T cs_6_6 -enable-16bit-types -HV 2021 -O3
-Vd -fspv-target-env=vulkan1.3 -fvk-t-shift 32 0 -fvk-u-shift 64 0`.
Patches: `_Static_assert` commented out in FNB_CT2D_ADD.hlsli;
`[unroll]`→`[loop]` in `build/hlsl_shadow` copies (JIT watchdog otherwise);
`-DDATA_TYPE=float...` (fp16 texture types invalid in Vulkan SPIR-V);
MSL `.xyzz`→`.xyzx` store swizzle fix (alpha=red on D3D12 semantics).
Native mpp kernels compile via `xcrun -f metal -std=metal4.0` at runtime
(cached `build/shaders/fnb_mpp.metallib` — delete after editing the .metal).
Shader caches in `build/shaders/fsr4_<key>` are NOT content-keyed —
`rm -rf` the dir after patch edits.

## Verified state

Accuracy vs WARP golden (quality case, M5): fp16 ~40.5–42.5 dB per frame
(f07: R41.7/G43.9/B42.5). Residual is ±1-LSB quantization-boundary flips
accumulating through the FNB chain — sign-symmetric, no spatial pattern.
int8 path: ~44–50 dB. `pre` is near bit-exact. On real HW output is
bit-stable across runs (zero self-noise); the VM has its own noise from
pass11 on (mean ~4.5e-4).

M5 perf (fp16 hot loop, same-session): per-pass sum ~21.3–23.4 ms/frame.
Per-pass table and bandwidth analysis in BENCHMARK.md — all passes sit at
8–38 GB/s of ~273 GB/s peak: nothing is memory-bound.

## Rejected approaches (with numbers — don't retry blindly)

- **INT8**: no DP4A on Apple GPUs; emulated unpack+fp32-mul-add. 120 ms/frame.
- **mpp::tensor_ops FNB** (`fnb_mpp.metal`, matmul2d for pw convs): microbench
  2099 vs 281 GFLOP/s did not transfer — skinny GEMMs (K,N = 32..128) + dw-conv
  staging overhead dominate. T=32 tiles: 13.4 vs 5.6 ms. T=128/64: 14.6 vs
  5.6 ms. Opt-in `--mpp`.
  API footgun: `matmul2d` produces exactly ONE 32×32 tile per `run()` —
  M>32 needs an in-kernel loop over 32-wide M slices.
- **Channel-parallel CNB** (`cnb_mpp`, simdgroup per pixel, K-split +
  `simd_shuffle_xor`): pass1 15.8 vs 1.99, pass2 13.9 vs 2.53, pass12 15.7 vs
  3.41 ms (4.6–7.9× slower). Opt-in `--cnb`. Do NOT roll out to FNB/9/11.
- **Barrier/encoder sync**: the 38.2 vs 23.4 ms "gap" was a cold-run
  measurement artifact — FUSED_FRAME equals per-pass sums (~21.2–21.5 ms).
  `barrier` is the default; `encoder` is not bit-identical (hazard reordering
  changes rounding order, f07 self-diff 50.6 dB, both ~42 dB vs WARP).
- **VM timings** are misleading per-pass: subgroup ops and matmul2d are
  emulated (~14 ms/pass for cnb_mpp vs ~3 generated). Use only for
  correctness. Also: `MTL_SHADER_VALIDATION=1` zeroes output on the VM and
  *perturbs* output on real HW — never diff goldens against validated runs.
  M5 GPU frequency scales: only same-session numbers are comparable.

## Remaining optimization levers (ranked)

Baseline after the 4-wide conv ILP, same-session M5 bench, quality 720→1080:
fused frame 19.8 ms → 13.3 ms (a repeat of the new build sat at 14.6 ms, so
treat the gain as about 5–6 ms). Per-pass after: post 2.22 (was 3.18),
pass1 1.32 (was 2.61), pass12 1.24 (was 2.92), pass2 1.30 (was 2.67),
pre 1.02 (was 1.15). Self-diff against the previous fp16 kernels is
~50–56 dB (mean abs 0.0003 on frame 0, 0.0017 by frame 3): same dot order,
but the wider inner step changes FMA contraction enough to flip requant
boundaries. It is not bit-identical to the previous fp16 build.

Done, do not revert blindly:

- **Live clear is on the GPU.** `fsr4_live.mm` no longer `memset`s scratch or
  `replaceRegion`s history/reproj/recurrent on the encode thread. Metal 4 uses
  `fillBuffer` plus a clear kernel; classic uses compute clears. First DLSS
  frame and every output-size change only. Steady frames are unchanged.
- **4-wide ILP inside the existing thread.** `fp16_ops.hlsli` accumulates four
  output channels per inner step in CNB, ConvNextBlock and FNB (same
  `f16_dot4` order). Post computes the four ConvTranspose taps together.
  `pre` dots four downscale features before the quad reduction. This is not
  the rejected lane-split (`--cnb`) or matmul2d (`--mpp`) path.

Still open, and the ceiling is small:

1. **Scratch layout** — NHWC half4 already; modest, frame is not memory-bound.
2. **weight residency** — tried via mpp. Constant-buffer weights already used
   where the source does it.
3. **dispatch coalescing** — encoder-split measured neutral.

Biggest structural unknown: the generated kernels assign one thread per
pixel×output-element with serial channel loops — proven better than both
naive lane-splitting and skinny-GEMM matmul2d. A hybrid (lane-parallel only
for the *wide* K stages like FNB contract 2NF→NF, keeping M tiles big) is
untested but the CNB result suggests shuffle-reduce overhead dominates at
FSR4's tiny channel counts.

## M5 benchmark protocol (exact commands)

On the user's M5, from the repo root:

```
make tools/bin/fsr4mac
# download golden-v0.1 release zips, unzip to a dir (say /path/tc)

# per-pass + fused frame, fp16 (default path)
./tools/bin/fsr4mac /path/tc/testcase_quality_720to1080 /tmp/o --fp16 --bench 10 --builddir build/shaders
# same but generated-only A/B for native kernels:
./tools/bin/fsr4mac /path/tc/testcase_quality_720to1080 /tmp/o --fp16 --bench 10 --builddir build/shaders   # base
./tools/bin/fsr4mac /path/tc/testcase_quality_720to1080 /tmp/o --fp16 --mpp --bench 10 --builddir build/shaders  # matmul2d FNB
./tools/bin/fsr4mac /path/tc/testcase_quality_720to1080 /tmp/o --fp16 --cnb --bench 10 --builddir build/shaders  # lane-parallel CNB

# correctness vs golden (writes fNN_out.bin to <out>)
./tools/bin/fsr4mac /path/tc/testcase_quality_720to1080 /tmp/o --fp16 --builddir build/shaders
./tools/bin/compare <golden>/final_q/f07_out.bin /tmp/o/f07_out.bin 1920 1080

# capability / tensor-ops probe
./tools/bin/fsr4mac --caps
./tools/bin/fsr4mac --bench-tensorops
```

Rules: compare only same-session numbers (GPU frequency scaling); never diff
`MTL_SHADER_VALIDATION` runs against goldens; VM numbers are correctness-only.

## Release/artifacts map

- GitHub Release `golden-v0.1`: testcase zips + README — correctness authority.
- Repo: `src/fsr4mac.mm` (host), `src/fnb_mpp.metal` (native kernels),
  `tools/gen_f16.py` (fp16 codegen + mpp manifest/weights),
  `tools/compile_variant.py` (DXC→SPIR-V→MSL pipeline),
  `patches/hlsl/**` (patched operator headers),
  `VERIFICATION.md` (accuracy evidence + rejected-approach metrics),
  `BENCHMARK.md` (perf protocol, per-pass table, bandwidth analysis),
  `README.md` (build/quickstart/toolchain details), `TESTDATA.md` (format).
