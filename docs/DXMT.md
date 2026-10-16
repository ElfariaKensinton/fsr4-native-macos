# DXMT integration

The previous M1 Pro kernel schedule change did not improve the reported in-game
FPS. It was a small isolated neural-kernel improvement, not the large gameplay
gain requested. The meaningful performance lever exposed here is the game's
DLSS render size: reducing it cuts the number of pixels the game shades before
FSR 4.1 Lite upscales to the requested output. Lower input sizes trade image
detail for speed; there is no honest universal FPS promise because games differ
in their GPU and CPU bottlenecks.

DXMT creates a MetalFX temporal scaler for its DLSS/NGX path. The integration
patch in [patches/dxmt/fsr4-dxmt-integration.patch](../patches/dxmt/fsr4-dxmt-integration.patch)
adds two hooks to DXMT:

- Load the FSR hook from `FSR4_DXMT_HOOK` before creating the first temporal
  scaler.
- Let the panel override the dimensions returned by the NGX optimal-settings
  callback. Install-time `FSR4_DXMT_RENDER_WIDTH` and
  `FSR4_DXMT_RENDER_HEIGHT` are defaults until the panel saves a size.

Apply the patch to a clean checkout of DXMT commit `50102f9`, then build and
install DXMT as usual:

```sh
git -C /path/to/dxmt checkout 50102f9
git -C /path/to/dxmt apply /path/to/fsr4-native-macos/patches/dxmt/fsr4-dxmt-integration.patch
# Build and install DXMT following its own instructions.
```

Configure FSR4 for the bottle:

```sh
python3 tools/crossover/crossover_fsr4.py install-dxmt --bottle "Game" \
  --profile lossy
```

The installer builds `dist/crossover/libfsr4hook.dylib` and the native
`dist/crossover/FSR4 Panel.app`, writes the hook path to the bottle, enables
`DXMT_ENABLE_NVEXT=1`, and configures the panel. It also removes a stale
`D3DMETAL_FRAMEWORK_PATH` override, which would make CrossOver launch the
D3DMetal route instead of DXMT. DXMT requires NVEXT for its NGX/DLSS
implementation. The selected profile is written to
`panel.conf`, replacing a stale panel choice that could otherwise override the
install-time profile. The installer does not copy or replace DXMT files. It
runs the x86_64 MetalFX swizzle smoke check and a live FSR replacement check
before saving the bottle settings. The hook is built for x86_64, matching the
CrossOver/Rosetta setup used by the existing D3DMetal integration.

The environment values printed by `install-dxmt` are the DXMT-specific bottle
configuration. `D3DMETAL_FRAMEWORK_PATH` is for the D3DMetal route and does not
load the hook into DXMT. Use the printed `FSR4_DXMT_HOOK`,
`DXMT_ENABLE_NVEXT`, `FSR4_ROOT`, and related values with a DXMT build that
includes the integration patch.

To choose an install-time default render size, pass both dimensions:

```sh
python3 tools/crossover/crossover_fsr4.py install-dxmt --bottle "Game" \
  --render-width 1280 --render-height 720 --profile lossy
```

The FSR4 Panel includes editable width and height fields plus one-click half,
40%, and one-third output presets based on the active output size. Set both
fields to `0` to use the game's selected preset. After applying, reopen the
game's DLSS setting so it requests dimensions again; some titles require
restarting the game. DXMT only returns these dimensions to titles that call
and honor its NGX optimal-settings callback. Each dimension must be at least
one third of the corresponding output dimension and no larger than output.
Invalid overrides are ignored.
Titles that choose render dimensions independently can ignore the override.

Reducing the internal render size can cut the game's pixel shading work by a
large amount, while image quality depends on the title, motion vectors, and
chosen scale. On M1–M4, FSR 4.1 Lite now reuses inference on alternating frames
up to 3x render scale, including Ultra Performance. An M1 Pro 2560×1600
synthetic same-device A/B fell from 24.371 ms to 15.243 ms of FSR GPU time
per frame; this is not a game-FPS measurement. See the
[M1 Pro benchmark report](reports/fsr411-m1pro-1600-2026-10-16.md).

Restore the bottle configuration with:

```sh
python3 tools/crossover/crossover_fsr4.py restore --bottle "Game"
```

The patch targets upstream DXMT commit
[50102f9](https://github.com/3Shain/dxmt/tree/50102f9). It has not been
compiled or runtime-validated in-game here. DXMT build instructions and current
installation details are maintained by [3Shain/dxmt](https://github.com/3Shain/dxmt).
