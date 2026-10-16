# DXMT integration

FSR 4.1.1 Lite selects the exact `simdgroup` INT8 backend on M1–M4. On an M1
Pro, an interleaved 8-round run at 1712×1112 output measured median full neural
frames at 11.94 ms with two simdgroups per Winograd tile versus 12.56 ms with
four (about 4.9% faster). A more aggressive one-simdgroup schedule was slower.
All eight output frames were byte-identical between the two- and four-group
schedules. The neural network, rounding, and temporal reuse cadence are
unchanged. This is a modest kernel-level gain, not a large end-to-end uplift;
the custom render-size option can reduce game shading work further, with the
usual image-quality trade-off at lower input resolutions.

DXMT creates a MetalFX temporal scaler for its DLSS/NGX path. The integration
patch in [patches/dxmt/fsr4-dxmt-integration.patch](../patches/dxmt/fsr4-dxmt-integration.patch)
adds two small opt-in hooks to DXMT:

- Load the FSR hook from `FSR4_DXMT_HOOK` before creating the first temporal
  scaler.
- Let `FSR4_DXMT_RENDER_WIDTH` and `FSR4_DXMT_RENDER_HEIGHT` override the
  dimensions returned by the NGX optimal-settings callback.

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

The installer builds `dist/crossover/libfsr4hook.dylib` and writes its path to
the bottle's `cxbottle.conf`. It does not copy or replace DXMT files. The hook
is currently built for x86_64, matching the CrossOver/Rosetta setup used by
the existing D3DMetal integration.

To force a custom internal resolution, pass both dimensions:

```sh
python3 tools/crossover/crossover_fsr4.py install-dxmt --bottle "Game" \
  --render-width 1280 --render-height 720 --profile lossy
```

DXMT returns these dimensions to titles that call and honor its NGX
optimal-settings callback. The values are required to be between one third of
the requested output size and the output size, matching the supported DLSS
scale interval. Invalid overrides are ignored by the DXMT patch. Games that
choose render dimensions independently can ignore them. Restart the bottle
after changing the configuration.

Reducing the internal render size can cut the game's pixel shading work by a
large amount, but image quality depends on the title, motion vectors, and
chosen scale. The FSR 4.1 Lite model, network weights, and temporal reuse
cadence are not changed by the override. Use the game's quality setting or
omit the override when preserving the highest input detail is more important
than GPU load.

Restore the bottle configuration with:

```sh
python3 tools/crossover/crossover_fsr4.py restore --bottle "Game"
```

The patch targets upstream DXMT commit
[50102f9](https://github.com/3Shain/dxmt/tree/50102f9). It has not been
compiled or runtime-validated here. DXMT build instructions and current
installation details are maintained by [3Shain/dxmt](https://github.com/3Shain/dxmt).
