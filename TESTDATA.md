# Test-data contract

Shared contract between the macOS Metal port (`fsr4mac`) and the Windows
DX12 golden-reference producer. Both sides must follow this byte-for-byte.

Ready-made golden testcases (DX12/WARP inputs + reference outputs + per-pass
dumps) are published as GitHub release
[`golden-v0.1`](https://github.com/UzenUPoZiTiV4ik/fsr4-mac/releases/tag/golden-v0.1)
— download `testcase_quality_720to1080.zip` and
`testcase_perf_540to1080.zip` to reproduce the verification in
VERIFICATION.md.

## Directory `testcase_<name>/`

`meta.json`:

```json
{
  "preset": "quality",
  "render_width": 1280, "render_height": 720,
  "upscale_width": 1920, "upscale_height": 1080,
  "frames": 8,
  "depth_inverted": false, "low_res_mv": true, "jittered_mv": false,
  "colorspace": "linear",
  "auto_exposure": false, "pre_exposure": 1.0,
  "rcas": false, "sharpness": 0.0,
  "frame": [
    { "jitter": [jx, jy], "mv_scale": [sx, sy], "reset": true,
      "frame_time_ms": 16.6, "camera_near": 0.1, "camera_far": 1000.0,
      "fovy": 1.0 }
  ]
}
```

- `preset`: native | quality | balanced | performance | ultraperf | drs
- `colorspace`: linear | srgb  (linear→ FFX_MLSR_COLORSPACE 0, srgb→ 2)
- `frame` has exactly `frames` entries.

## Per-frame input files (`NN` = zero-padded frame index)

| file | contents | size |
|---|---|---|
| `fNN_color.bin`  | RGBA16F linear colour, row-major, top-left origin | rw*rh*8 B |
| `fNN_depth.bin`  | float32 NDC depth in [0,1] | rw*rh*4 B |
| `fNN_mv.bin`     | RG16F motion vectors | rw*rh*4 B |
| `fNN_exposure.bin` | float32, optional | 4 B |

### Motion-vector convention

`mv.bin` stores, per render pixel, **the position of this pixel's surface in
the previous frame minus its position in the current frame, in render-pixel
units**:

    mv = prev_px - cur_px

`mv_scale` in meta.json is the `FfxApiFloatCoords2 motionVectorScale` value
(units→render-pixels); our generator uses `[1,1]` since mv is already in
pixels. The host computes `main.mv_scale = mv_scale / mvSize` where
`mvSize = renderSize` when `low_res_mv` (else upscaleSize) — exactly as the
provider does. Contract testcases always use `low_res_mv: true`.

### Jitter

`jitter` is the subpixel offset (pixels, relative to pixel centre) applied
when the frame was rendered — Halton(2,3), 8 phases: `halton(i%8+1, 2)-0.5`,
`halton(i%8+1, 3)-0.5`. `gen_testcase` samples the procedural texture at the
jittered position, matching a real jittered render.

## Outputs

| file | contents |
|---|---|
| `fNN_out.bin`  | RGBA16F, `upscale_w * upscale_h * 8` B, row-major top-left |
| `fNN_out.png`  | 8-bit RGBA for viewing (linear→sRGB applied when colorspace=linear) |
| `fNN_passPP_scratch.bin` | raw scratch buffer dump after model pass PP (`--dump-passes`) |
| `fNN_recurrent.bin` | recurrent RGBA8 texture dump |
| `fNN_history.bin`    | history RGBA16F texture dump |

## Generator

    tools/bin/gen_testcase <dir> <preset> <rw> <rh> <uw> <uh> <frames>

Deterministic (plain float math, no RNG): infinite procedural plane
(checkerboard + gradient + high-frequency stripes), camera translating
(1.75, 0.50) render-px/frame, one square object counter-translating
(-2.25, +1.25) px/frame at closer depth. Depth is a vertical planar
gradient 0.85→0.15 (top→bottom), object at 0.12. Frame 0 has reset=true.
Bit-identical output on any platform.

Defaults: `testcase_quality_720to1080` (quality, 1280x720→1920x1080, 8
frames) and `testcase_perf_540to1080` (performance, 960x540→1920x1080, 8).

## Compare

    tools/bin/compare <a.bin> <b.bin> <w> <h> [diff.png]
    tools/bin/compare <dirA> <dirB> [diff_dir]     # compares all fNN_out.bin

Per-channel max-abs / mean-abs / PSNR; diff PNG amplifies |a-b| x16.
