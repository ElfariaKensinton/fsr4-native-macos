#!/usr/bin/env python3
"""Point one CrossOver bottle at a copied D3DMetal that loads libfsr4hook.

The CrossOver.app bundle is not modified. libd3dshared reads
D3DMETAL_FRAMEWORK_PATH and dlopens that path (RTLD_NOW). Restore deletes
only that key from the bottle config.
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist" / "crossover"
HOOK_SRC = ROOT / "hook" / "fsr4_metalfx.m"
LIVE_SRC = ROOT / "src" / "fsr4_live.mm"
LIVE411_SRC = ROOT / "src" / "fsr411_live.mm"
PANEL_SRC = ROOT / "src" / "fsr4_panel.mm"          # hook side: status publish + app launch
PANEL_APP_SRC = ROOT / "tools" / "crossover" / "panel_app.mm"
PANEL_APP = DIST / "FSR4 Panel.app"
SMOKE_SRC = ROOT / "tools" / "crossover" / "smoke.m"
LIVE_TEST_SRC = ROOT / "tools" / "crossover" / "live_test.mm"
DLOPEN_SRC = ROOT / "tools" / "crossover" / "dlopen_test.c"
# D3DMetal inside a CrossOver.app, in order of preference. Newer builds also
# ship GPTK 3 in apple_gptk3; older ones only have apple_gptk.
FRAMEWORKS_IN_APP = ("Contents/SharedSupport/CrossOver/lib/apple_gptk/external/D3DMetal.framework",
                     "Contents/SharedSupport/CrossOver/lib/apple_gptk3/external/D3DMetal.framework")
FRAMEWORK_IN_APP = FRAMEWORKS_IN_APP[0]
CROSSOVER_MARKER = "Contents/SharedSupport/CrossOver"
CROSSOVER_BUNDLE_ID = "com.codeweavers.CrossOver"
APP_FOLDERS = (Path("/Applications"), Path.home() / "Applications")
STANDARD_NAMES = ("CrossOver.app", "CrossOver Preview.app")
BOTTLES = Path.home() / "Library/Application Support/CrossOver/Bottles"
# Same deployment target as the Makefile: the live bridge uses Metal 4 APIs.
MIN_MACOS = "26.0"
FSR402_WEIGHTS = ROOT / "third_party/fsr4/internal/shaders/fsr4_model_v07_i8_quality/initializers.bin"
LOAD_PATH = b"@loader_path/libfsr4hook.dylib"
LC_LOAD_DYLIB = 0xC
KEY = "D3DMETAL_FRAMEWORK_PATH"
LOG_KEY = "FSR4_HOOK_LOG"
MARKER = ";; fsr4-mac"
ROOT_KEY = "FSR4_ROOT"
REPLACE_KEY = "FSR4_REPLACE"
PANEL_KEY = "FSR4_PANEL"
VERSION_KEY = "FSR4_VERSION"
PROFILE_KEY = "FSR4_PROFILE"
METALFX_KEY = "D3DM_ENABLE_METALFX"
DXMT_HOOK_KEY = "FSR4_DXMT_HOOK"
DXMT_RENDER_WIDTH_KEY = "FSR4_DXMT_RENDER_WIDTH"
DXMT_RENDER_HEIGHT_KEY = "FSR4_DXMT_RENDER_HEIGHT"
# "fsr4": the FSR 4.0.2 model under a 4.1.1 install (FSR4_PROFILE only, not in the panel).
PROFILES = ("exact", "exact-fast", "lossy", "fsr4")
OURS = (KEY, LOG_KEY, ROOT_KEY, REPLACE_KEY, PANEL_KEY, VERSION_KEY, PROFILE_KEY,
        DXMT_HOOK_KEY, DXMT_RENDER_WIDTH_KEY, DXMT_RENDER_HEIGHT_KEY)


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, **kw)


def sdk():
    return subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()


def build_hook(log_path: Path) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / "libfsr4hook.dylib"
    common = ["-O2", "-fobjc-arc", "-arch", "x86_64", f"-mmacosx-version-min={MIN_MACOS}",
              "-isysroot", sdk(), "-I", str(ROOT / "src")]
    run(["clang", "-c", *common, str(HOOK_SRC), "-o", str(DIST / "hook.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(LIVE_SRC), "-o", str(DIST / "live.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(LIVE411_SRC), "-o", str(DIST / "live411.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(PANEL_SRC), "-o", str(DIST / "panel.o")])
    run([
        "clang++", "-dynamiclib", *common,
        str(DIST / "hook.o"), str(DIST / "live.o"), str(DIST / "live411.o"), str(DIST / "panel.o"),
        "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
        # Darwin ld needs an explicit allowance as well as weak_import when
        # linking an encoder without the optional 4.1.1 symbol. The public
        # version router rejects an unavailable requested model explicitly.
        "-Wl,-U,_fsr411_encode_live_frame",
        "-Wl,-U,_fsr411_invalidate_owner", "-Wl,-U,_fsr411_release_owner",
        "-install_name", "@loader_path/libfsr4hook.dylib",
        "-o", str(out),
    ])
    run(["codesign", "-f", "-s", "-", str(out)])
    return out


def build_smoke() -> Path:
    out = DIST / "smoke"
    run([
        "clang", "-O2", "-fobjc-arc", "-arch", "x86_64",
        f"-mmacosx-version-min={MIN_MACOS}", "-isysroot", sdk(),
        str(SMOKE_SRC), str(DIST / "libfsr4hook.dylib"),
        "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
        "-Wl,-rpath", str(DIST),
        "-o", str(out),
    ])
    return out


def build_live() -> Path:
    out = DIST / "live_test"
    run([
        "clang++", "-O2", "-fobjc-arc", "-std=c++17", "-arch", "x86_64",
        f"-mmacosx-version-min={MIN_MACOS}", "-isysroot", sdk(),
        "-I", str(ROOT / "src"),
        str(LIVE_TEST_SRC), str(DIST / "libfsr4hook.dylib"),
        "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
        "-Wl,-rpath", str(DIST),
        "-o", str(out),
    ])
    return out


def build_panel_app() -> Path:
    """dist/crossover/FSR4 Panel.app: native (arm64) Dock app showing the hook's
    status and setting the RCAS sharpness (tools/crossover/panel_app.mm)"""
    macos = PANEL_APP / "Contents" / "MacOS"
    res = PANEL_APP / "Contents" / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    exe = macos / "FSR4 Panel"
    run([
        "clang++", "-O2", "-fobjc-arc", "-std=c++17", "-arch", "arm64",
        f"-mmacosx-version-min={MIN_MACOS}", "-isysroot", sdk(),
        str(PANEL_APP_SRC),
        "-framework", "Foundation", "-framework", "AppKit", "-framework", "QuartzCore",
        "-o", str(exe),
    ])
    (PANEL_APP / "Contents" / "Info.plist").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>FSR 4</string>
  <key>CFBundleDisplayName</key><string>FSR 4</string>
  <key>CFBundleExecutable</key><string>FSR4 Panel</string>
  <key>CFBundleIdentifier</key><string>org.fsr4mac.panel</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>LSMinimumSystemVersion</key><string>{MIN_MACOS}</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
""")
    iconset = DIST / "icon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()
    run([str(exe), "--make-icon", str(iconset)])
    run(["iconutil", "-c", "icns", str(iconset), "-o", str(res / "icon.icns")])
    shutil.rmtree(iconset)
    run(["codesign", "-f", "-s", "-", str(PANEL_APP)])
    return PANEL_APP


def build_dlopen() -> Path:
    out = DIST / "dlopen_test"
    run([
        "clang", "-O2", "-arch", "x86_64", f"-mmacosx-version-min={MIN_MACOS}",
        "-isysroot", sdk(), str(DLOPEN_SRC), "-o", str(out),
    ])
    return out


def patch_macho(path: Path):
    data = bytearray(path.read_bytes())
    if LOAD_PATH in data:
        print("load command already present")
        return
    magic, cputype, cpusub, filetype, ncmds, sizeofcmds, flags, reserved = struct.unpack_from("<IIIIIIII", data, 0)
    if magic != 0xFEEDFACF:
        raise SystemExit(f"not a 64-bit mach-o: {path}")
    cmds_end = 32 + sizeofcmds
    # name is 8-aligned inside the command; 24-byte header + path
    name = LOAD_PATH + b"\0"
    pad = (8 - (len(name) % 8)) % 8
    name += b"\0" * pad
    cmdsize = 24 + len(name)
    if cmdsize % 8:
        raise SystemExit("cmdsize alignment")
    # do not walk into __text
    text_off = None
    off = 32
    for _ in range(ncmds):
        cmd, csz = struct.unpack_from("<II", data, off)
        if cmd == 0x19:
            nsects = struct.unpack_from("<I", data, off + 64)[0]
    