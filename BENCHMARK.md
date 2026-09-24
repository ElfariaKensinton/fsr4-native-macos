# BENCHMARK.md — how to benchmark fsr4mac

## 2026-09-24: portable backend selection and FP16 boundaries

`auto` is now the default in the CLI and live encoder: generated FP16 on
M1–M4, native INT8 on compatible M5-class GPUs. Pro/Max/Ultra follow their
Metal feature family. See [HARDWARE.md](docs/HARDWARE.md) for capability
checks, fallback, installation and the limits of physical validation.

The FP16 change removes invalid work at tensor/frame edges. Previously,
FNB threads past the end of 480- and 240-pixel rows raced valid writers in
the next row, producing different left-edge pixels on repeated runs.
Bounds checks fix this and skip those calculations; post also skips
off-frame 2x2 blocks before evaluating the network.

Physical M5, Low Power Mode off, `--fp16 --bench 12 --reps 10`, three
alternating A/B runs per case (24 fused frames/run), medians of means:

| testcase | before | after | reduction |
|---|---:|---:|---:|
| quality 720 → 1080 | 12.66 ms | 12.56 ms | 0.7% |
| performance 540 → 1080 | 12.78 ms | 12.60 ms | 1.4% |
| balanced 1000x648 → 1712x1112 | 12.18 ms | 12.09 ms | 0.7% |

The speed gain is small and sensitive to GPU clocks. The principal result
is deterministic, correct boundary handling. No large FP16 speedup is
claimed for M1–M4 without measurements on those chips.

Additional candidates were measured and discarded: FP16 post colour caching
(post roughly unchanged around 1.75 ms), fully unrolled channel reductions
(over two minutes in the driver compiler before cancellation), paired channel
reductions (about 13.7 vs 11.5 ms/frame), native scalar/four-thread CNB kernels
(1.53/1.82 vs about 1.2 ms/pass), 128-thread groups and pipeline sizing hints
(no consistent frame-time improvement).

Validation: 72 capability-policy checks, five Python tests, GPU checks of
automatic selection and forced native failure, repeated byte-identical FP16
outputs, and six classic/Metal 4 live tests under Rosetta. All 55 frames in
the 15-scenario FP16 suite matched the version with corrected tensor bounds
before the post early exit. Native M5 output remains unchanged. Existing
SPD auto-exposure nondeterminism on the small partial-tile testcase remains;
initializing the CLI exposure texture, as the live encoder already does,
does not resolve it.

## 2026-09-24 follow-up: parallel post filter

Baseline: the already optimized colour-cache/CT2D version in the next
section. Same physical M5, Low Power Mode off, three alternating A/B runs
per case with `--i8mpp --bench 20 --reps 10` (40 fused frames/run). Medians
of the three means:

| testcase | previous optimization | this version | additional GPU time reduction |
|---|---:|---:|---:|
| quality, 1280x720 → 1920x1080 | 4.83 ms | 4.67 ms | 3.2% |
| performance, 960x540 → 1920x1080 | 4.86 ms | 4.63 ms | 4.7% |
| balanced, 1000x648 → 1712x1112 | 4.48 ms | 4.33 ms | 3.3% |

Compare within this series: GPU clock variation accounts for the different
absolute baseline times versus the earlier measurements below.

`post_filter.hlsl` assigns one output pixel to each thread in a 16x16
group, replacing the serial four-pixel loop in an 8x8 group. The host still
dispatches one group per 16x16 output tile and reads the actual threadgroup
size from shader metadata. Cache bounds are checked once for the full 3x3
footprint, rather than for each tap. Quality's post stage fell from 0.514
to 0.362 ms (about 30%); performance/balanced measured 0.512 → 0.356 and
0.465 → 0.330 ms. Native matrix operands now expose their fixed tile extents,
and requantization narrows a saturated `int4` directly to a packed `char4`.
For example, quality's pass 1 fell from 0.381 to 0.366 ms.

The same 15-scenario, 55-frame suite below passed again with byte-identical
outputs. Fixed-input post replays at all six exposure values also matched
output and history with Metal API validation enabled. The x86_64 live
Metal 4/Rosetta test matched the CLI exactly. Auto-exposure's previously
observed full-pipeline nondeterminism remains outside this equality claim.

Rejected experiments: cooperative register accumulators with scalar
requantization (CNB about 0.45 vs 0.37 ms); a pre-stage colour cache
(about 0.73 vs 0.68 ms, also changed rounding); four-feature pre-stage
loops (no consistent gain); and filling a smaller post cache based on the
scale (about 0.37 vs 0.35 ms). The selected post source uses an angle-bracket
include so DXC uses the shadow tree's loop policy. A quoted local include
accidentally enabled unrolling and changed the output; that variant was
discarded.

`tools/precompile_i8mpp.sh` has been rerun for the live shader variants.
Restart the game to pick up the new pipelines.

## 2026-09-24: post-filter cache and packed CT2D taps

Physical Apple M5, macOS 27.0 (26A428), Low Power Mode off on both AC and
battery. Baseline: `702b9c7`. Both versions used `--i8mpp --bench 20 --reps 10`
with the same recorded inputs, barriers and generated pre stage. Three
alternating A/B runs per testcase, reversing order on the middle run;
numbers below are medians of the three fused-frame means (40 frames/run).

| testcase | before | after | GPU time reduction |
|---|---:|---:|---:|
| quality, 1280x720 → 1920x1080 | 5.69 ms | 5.10 ms | 10.3% |
| performance, 960x540 → 1920x1080 | 5.57 ms | 4.98 ms | 10.5% |
| balanced, 1000x648 → 1712x1112 | 5.31 ms | 4.64 ms | 12.5% |

These are upscaler GPU times, not total game-frame times or game FPS.
Old 10+ ms results obtained in Low Power Mode should not be used for this
comparison. GPU clock variation still affects individual runs.

The post filter reuses float32 colours transformed with `ApplyMuLaw`
within each threadgroup. CT2D passes 9/11/13 issue one wider matrix multiply
for all four taps, without increasing accumulator storage. On quality,
the isolated post stage fell from about 0.88 to 0.53 ms, pass 9 from 0.35
to 0.28 ms, and pass 13 from 0.60 to 0.55 ms. Pass 11 was roughly neutral
(0.43 → 0.44 ms); the frame improvement is measured across the entire chain.

Validation against the previous Metal version (not a new DX12 golden run):

- **55/55 final frames are byte-identical** across 15 scenarios: the three
  recorded cases (2 frames each); 8-frame quality and performance sequences;
  an 8-frame balanced 421x243 → 641x365 sequence; native, ultraperf and DRS
  with 321x181 output (3 each); DRS 384x240 → 320x192 (3); sRGB, PQ and
  RCAS 0.4 variants of the odd-size case (3 each); quality 3840x2160 (2);
  and a 3856x24 case exercising the 4320 shader tier (2).
- The recorded quality case also matched all per-pass scratch dumps and
  recurrent/history state on both frames.
- The small auto-exposure testcase was **not deterministic in the baseline**:
  baseline-versus-baseline reruns already differed, starting before the
  modified model passes. It cannot establish full-frame equality. Replaying
  identical scratch inputs through passes 9/11/13 gave identical scratch
  and recurrent data; replaying the post stage with fixed exposure values
  0.03, 0.25, 0.7, 1, 2 and 8 gave identical output and history. The existing
  auto-exposure instability is not fixed by this optimization.
- The live Metal 4 encoder compiled for x86_64 and ran under Rosetta with
  output identical to the CLI (`live_test ... mtl4`, sharpness 0).

Refresh the game shader cache with `tools/precompile_i8mpp.sh` and restart
the game. The CLI rebuilds its shader variants automatically. Native kernel
entry points include `_tap4` to reject manifests with the old CT2D layout.

## Build

```
make
./build_shaders.sh        # compiles all presets/res (needs dxc + spirv-cross, see README)
```

Shaders are also JIT-compiled on demand by `fsr4mac` itself — a bare
`fsr4mac <testcase> <out> [--fp16]` works without `build_shaders.sh`.

## Flags

| flag | meaning |
|---|---|
| `--auto` | Default. Native INT8 on compatible M5-class GPUs, FP16 on older GPUs or after native initialization failure. |
| `--int8` | INT8 model path (original, DP4A-emulated). Explicit reference mode. |
| `--fp16` | FP16 model path (offline-dequantized weights, `gen_f16.py` + `fp16_ops.hlsli`). |
| `--i8mpp` | INT8 model passes as native Metal 4 tensor-ops kernels on the M5 neural accelerators (`src/i8mpp.metal`, `tools/gen_i8mpp.py`); post's ConvNextBlock + ConvTranspose included (pass 13, `_pt` post variant). Reference int8 numerics. Fastest. |
| `--bench N` | Run N iterations per frame over all 8 frames, print per-stage GPU ms and TOTAL. Skips PNG/bin writes and pass dumps. |
| `--reps R` | With `--bench`: encode each stage R times per command buffer and report the mean per dispatch (warm GPU clock; per-stage numbers are ±30% noisy with R=1). `--bench 3 --reps 10` is the recommended per-stage protocol. |
| `--dump-passes` | Dump the scratch buffer after every stage (`fNN_passPP_scratch.bin`) + constants + reprojected history. |
| `--caps` | Print device name, GPU families, Metal 4 runtime presence, and a functional `mpp::tensor_ops::matmul2d` compile+pipeline probe. |
| `--bench-tensorops` | Dispatch + time a 1024x1024x4096 fp16 GEMM via `mpp::tensor_ops::matmul2d` vs `simdgroup_matrix` (both verified vs a CPU ref first); prints GFLOP/s per path. Decides which MMA the convs should use. |
| `--builddir DIR` | Shader build dir (default `build/shaders`). |

## Benchmark recipe (on the M5)

```
./tools/bin/fsr4mac --caps                                  # device + matmul2d compile probe
./tools/bin/fsr4mac --bench-tensorops                       # timed matmul2d vs simdgroup GEMM
./tools/bin/fsr4mac testcases/testcase_quality_720to1080 /tmp/b_i8 --int8 --bench 20
./tools/bin/fsr4mac testcases/testcase_quality_720to1080 /tmp/b_f16 --fp16 --bench 20
```

Report the `TOTAL(per-pass sum)` line plus the per-stage table back.

## `--mpp` (opt-in)

Native `mpp::tensor_ops` matmul2d FNB kernels (`src/fnb_mpp.metal`,
manifest+transposed weights from gen_f16.py). **Measured slower than the
generated fp16 kernels on M5** (see VERIFICATION.md for both iterations) —
opt-in via `--mpp`, default off. Bench CSV stages get `_mpp` suffix.

## `--sync` modes

Inter-pass sync strategy (model passes only touch the scratch buffer; pre/post
also touch textures):

| mode | mechanism |
|---|---|
| `barrier` (default) | `memoryBarrierWithScope` after every dispatch — `Buffers` for model passes, `Buffers|Textures` for pre/post/spd/rcas |
| `encoder` | new compute encoder between passes in the same command buffer — driver hazard tracking inserts only needed fences |
| `none` | no sync — wrong results, upper-bound measurement only |

`--bench` reports both per-stage CB timings and a fused single-command-buffer
frame time (`FUSED_FRAME_<sync>` csv line + `fused frame gpu_ms`) so the
per-stage-sum vs fused gap is visible in one run.

GPU frequency scaling on M5 causes large run-to-run variation (observed int8
120->61 ms at identical binary/cached shaders) — only compare numbers from
the same run/session.

## M5 hotspot analysis (fp16, quality 720->1080, per-pass table)

Bytes moved vs measured ms vs ~273 GB/s — every pass is at 8-38 GB/s
(3-14% of peak): all passes are latency/compute-bound, NOT memory-bound.
Fusion/layout changes would not help; the generated kernels' serial
per-thread channel loops (each thread does a whole pixel's channels) are
the bottleneck. Lever: redistribute [pixel,channel] work across threads.

| pass | MB moved | M5 ms | GB/s |
|---|---|---|---|
| pre | ~64 | 1.77 | ~36 |
| pass1/2/12 CNB enc0 | 66 | 3.10/2.56/3.37 | 20-26 |
| pass4/5/10 FNB32 | 17 | ~1.15 | ~14 |
| pass7/8 FNB64 | 8 | ~1.05 | ~8 |
| post | ~83 | 3.50 | ~24 |

## Notes / caveats

- GPU timestamps come from `MTLCommandBuffer` `GPUStartTime`/`GPUEndTime` per
  dispatch. On the VirtualMac paravirtual GPU absolute numbers are unreliable
  (host calls are timed, not real silicon); treat them as directional only.
  The M5 is the perf authority.
- `--bench` keeps the full per-frame dependency chain (recurrent state), so it
  measures a realistic frame, not isolated kernels.
- Baseline measured on physical M5 @ 720→1080, int8 path: ~120 ms/frame.
- On this VM (VirtualMac2.1 paravirtual GPU, macOS 26.5.2): int8 ≈ 107 ms,
  fp16 ≈ 45 ms per frame (directional only).
- `--bench-tensorops` is the decisive check: on this VM matmul2d already
  measures 2.4 TFLOP/s vs simdgroup_matrix 0.64 TFLOP/s (both verified correct
  on a small GEMM first). If matmul2d >> simdgroup on the M5 too, the convs
  get ported to tensor_ops; parity means emulation and simdgroup_matrix wins.

## i8mpp (M5, 2026-09-23, same session)

```
./tools/bin/fsr4mac testcases/testcase_quality_720to1080 /tmp/o --i8mpp --bench 3 --reps 10 --builddir build/shaders
```

| case | int8 | fp16 | i8mpp (CNB-only pass 13) | i8mpp (CNB+CT2D pass 13), mean / min |
|---|---|---|---|---|
| quality 720->1080, fused frame | 47.5 ms | 12.7 ms | 5.95–6.3 ms | **5.57 / 4.97 ms** |
| performance 540->1080, fused frame | — | — | — | 5.41 / 4.95 ms |
| balanced 1000x648->1712x1112, fused frame | — | 11.1 ms | 5.8 ms | 5.06 / 4.74 ms |

Per-stage (quality 1080, `--bench 10 --reps 10`): pass1/2/12 0.37–0.41 ms,
pass3/6 0.06–0.08, pass4/5/10 0.22–0.24, pass7/8 0.16–0.17, pass9 0.34,
pass11 0.42, pass13 (post CNB+CT2D) 0.61, post 0.86, pre 0.68 (sum 5.20). Accuracy vs DX12 golden: quality 47.1/44.0/41.9 dB (f00/f03/f07),
perf 48.7/42.7/41.9 dB; model passes 1–9 bit-exact against the golden
per-pass dumps on frame 0 (frame 1: 4/7/10 have 1–3 single-LSB flips). Single passes: `tools/bench/i8_test <gendir>
<dumpdir> 0 <pass> 5`.

## Machine results log

| machine | chip/os | tool | result |
|---|---|---|---|
| user's machine | Apple M5, macOS 27.0 | `--bench` quality 720→1080 | int8 120 ms/frame, fp16 43.2 ms (2.78x); post 14.6, pass12 4.5, pass2 3.6, pass1 3.1, pre 2.8, other convs 0.6–2.4; whole-CB mean 44.9 ms |
| user's machine | M5 | fp16 vs golden | f07 R41.57/G43.75/B42.48 dB; f00 R 39.28 dB; per-frame 40.5–42.5 dB |
| user's machine | M5, macOS 27.0 | `--bench-tensorops` | matmul2d 2099 GFLOP/s vs simdgroup 281 GFLOP/s (7.5x) — neural accelerator path confirmed |
| user's machine | M5 | fp16 `--bench` post-fix | per-stage sum 23.4 ms (post 3.5), fused CB 38.2 ms — ~15 ms dispatch/sync overhead under investigation |
| this VM | VirtualMac2.1 paravirtual, macOS 26.5.2 | `--bench` | int8 107 ms, fp16 45.2 ms per-pass sum |
| this VM | paravirtual | `--bench-tensorops` | matmul2d 2388 GFLOP/s, simdgroup_matrix 639 GFLOP/s |

## Lane-parallel CNB (`cnb_mpp` kernel)

`src/fnb_mpp.metal` `cnb_mpp`: one 32-lane simdgroup per output pixel
(4 pixels / 128-thread threadgroup). conv1 3x3 (16->16): lanes split
K=8+8, `simd_shuffle_xor(16)` reduce. conv2 1x1 (16->32): lane = output,
dot over 16. conv3 1x1 (32->16): lanes split K=16+16, xor reduce,
bias + residual + requant. Requant semantics identical to the generated
`f16_ConvNextBlock`. Applies to passes 1/2/12 (CNB16 shape guard in the
emitter; other CNB shapes stay on the generated kernel).

M5 result (`9457a84`): lane-parallel loses 4.6-7.9x — pass1 15.8 vs
1.99, pass2 13.9 vs 2.53, pass12 15.7 vs 3.41 ms; frame 58.7 vs 22.8.
The generated kernel's serial channel loops beat the split-K + shuffle
reduction: 32x more threads at ~1/32 work each exposes launch/shuffle
overhead, and the stride-scattered weight reads do not coalesce. Do NOT
roll out to FNB/9/11. Kept behind `--cnb` as a documented dead end.
