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
PANEL_SRC = ROOT / "src" / "fsr4_panel.mm"          # hook side: status publish + app launch
PANEL_APP_SRC = ROOT / "tools" / "crossover" / "panel_app.mm"
PANEL_APP = DIST / "FSR4 Panel.app"
SMOKE_SRC = ROOT / "tools" / "crossover" / "smoke.m"
LIVE_TEST_SRC = ROOT / "tools" / "crossover" / "live_test.mm"
DLOPEN_SRC = ROOT / "tools" / "crossover" / "dlopen_test.c"
SRC_FRAMEWORK = Path(
    "/Applications/CrossOver Preview.app/Contents/SharedSupport/CrossOver"
    "/lib/apple_gptk/external/D3DMetal.framework"
)
DEFAULT_BOTTLE = Path.home() / "Library/Application Support/CrossOver/Bottles/Игры/cxbottle.conf"
LOAD_PATH = b"@loader_path/libfsr4hook.dylib"
LC_LOAD_DYLIB = 0xC
KEY = "D3DMETAL_FRAMEWORK_PATH"
LOG_KEY = "FSR4_HOOK_LOG"
MARKER = ";; fsr4-mac"
ROOT_KEY = "FSR4_ROOT"
REPLACE_KEY = "FSR4_REPLACE"
PANEL_KEY = "FSR4_PANEL"
OURS = (KEY, LOG_KEY, ROOT_KEY, REPLACE_KEY, PANEL_KEY)


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, **kw)


def sdk():
    return subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()


def build_hook(log_path: Path) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / "libfsr4hook.dylib"
    common = ["-O2", "-fobjc-arc", "-arch", "x86_64", "-mmacosx-version-min=14.0",
              "-isysroot", sdk(), "-I", str(ROOT / "src")]
    run(["clang", "-c", *common, str(HOOK_SRC), "-o", str(DIST / "hook.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(LIVE_SRC), "-o", str(DIST / "live.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(PANEL_SRC), "-o", str(DIST / "panel.o")])
    run([
        "clang++", "-dynamiclib", *common,
        str(DIST / "hook.o"), str(DIST / "live.o"), str(DIST / "panel.o"),
        "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
        "-install_name", "@loader_path/libfsr4hook.dylib",
        "-o", str(out),
    ])
    run(["codesign", "-f", "-s", "-", str(out)])
    return out


def build_smoke() -> Path:
    out = DIST / "smoke"
    run([
        "clang", "-O2", "-fobjc-arc", "-arch", "x86_64",
        "-mmacosx-version-min=14.0", "-isysroot", sdk(),
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
        "-mmacosx-version-min=14.0", "-isysroot", sdk(),
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
        "-mmacosx-version-min=14.0", "-isysroot", sdk(),
        str(PANEL_APP_SRC),
        "-framework", "Foundation", "-framework", "AppKit", "-framework", "QuartzCore",
        "-o", str(exe),
    ])
    (PANEL_APP / "Contents" / "Info.plist").write_text("""<?xml version="1.0" encoding="UTF-8"?>
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
  <key>LSMinimumSystemVersion</key><string>14.0</string>
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
        "clang", "-O2", "-arch", "x86_64", "-mmacosx-version-min=14.0",
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
            so = off + 72
            for _s in range(nsects):
                sect = data[so:so + 16].split(b"\0", 1)[0]
                foff = struct.unpack_from("<I", data, so + 48)[0]
                if sect == b"__text":
                    text_off = foff
                so += 80
        off += csz
    if text_off is None:
        raise SystemExit("no __text")
    if cmds_end + cmdsize > text_off:
        raise SystemExit(f"no header pad: need {cmdsize}, have {text_off - cmds_end}")
    if any(data[cmds_end:cmds_end + cmdsize]):
        raise SystemExit("header pad is not zero")
    struct.pack_into("<IIIIII", data, cmds_end, LC_LOAD_DYLIB, cmdsize, 24, 0, 0x10000, 0x10000)
    data[cmds_end + 24:cmds_end + 24 + len(name)] = name
    struct.pack_into("<II", data, 16, ncmds + 1, sizeofcmds + cmdsize)
    path.write_bytes(data)
    print(f"added LC_LOAD_DYLIB {LOAD_PATH.decode()} cmdsize={cmdsize}")


def copy_framework(hook: Path) -> Path:
    dest = DIST / "D3DMetal.framework"
    if dest.exists():
        shutil.rmtree(dest)
    if not SRC_FRAMEWORK.is_dir():
        raise SystemExit(f"missing {SRC_FRAMEWORK}")
    shutil.copytree(SRC_FRAMEWORK, dest, symlinks=True)
    binary = dest / "Versions" / "A" / "D3DMetal"
    patch_macho(binary)
    shutil.copy2(hook, dest / "Versions" / "A" / "libfsr4hook.dylib")
    run(["codesign", "-f", "-s", "-", str(binary)])
    run(["codesign", "-f", "-s", "-", str(dest / "Versions" / "A" / "libfsr4hook.dylib")])
    return binary


def conf_set(conf: Path, values: dict):
    text = conf.read_text()
    lines = text.splitlines()
    seen = set()
    out = []
    for raw in lines:
        hit = None
        for key in values:
            if raw.strip().startswith(f'"{key}"'):
                hit = key
                break
        if hit:
            out.append(f'"{hit}" = "{values[hit]}"')
            seen.add(hit)
        else:
            out.append(raw)
    missing = [k for k in values if k not in seen]
    if missing:
        if MARKER not in out:
            out.append(MARKER)
        for key in missing:
            out.append(f'"{key}" = "{values[key]}"')
    bak = DIST / "backups" / (conf.parent.name + ".cxbottle.conf")
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists():
        shutil.copy2(conf, bak)
        print(f"backup {bak}")
    conf.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""))
    print(f"set {', '.join(values)} in {conf}")


def conf_unset(conf: Path):
    if not conf.exists():
        print(f"no bottle config {conf}")
        return
    lines = conf.read_text().splitlines()
    out = []
    for raw in lines:
        s = raw.strip()
        if s == MARKER or any(s.startswith(f'"{key}"') for key in OURS):
            continue
        out.append(raw)
    conf.write_text("\n".join(out) + "\n")
    print(f"removed {', '.join(OURS)} from {conf}")


def build_reference(env):
    # Build a reference using this device's backend. A saved M5 INT8 frame
    # is not the FP16 reference that M1-M4 need for the installation test.
    run(["make", "-B", "tools/bin/fsr4mac"], cwd=ROOT)
    model = env.get("FSR4_MODEL", "auto")
    if model == "int8":
        model = "i8mpp"  # historic live alias, not CLI reference INT8
    if model not in ("auto", "f16", "fp16", "i8mpp"):
        model = "auto"
    info = json.loads(subprocess.check_output(
        [str(ROOT / "tools/bin/fsr4mac"), "--backend-info", model], text=True, env=env))
    backend = info["backend"]
    print(f"{info['device']}: {backend} ({info['reason']})")
    testcase = ROOT / "testcases" / "testcase_quality_720to1080"
    if not (testcase / "meta.json").exists():
        run(["make", "tools/bin/gen_testcase"], cwd=ROOT)
        run([str(ROOT / "tools/bin/gen_testcase"), str(testcase),
             "quality", "1280", "720", "1920", "1080", "2"])
    refdir = DIST / ("reference-" + backend)
    run([str(ROOT / "tools/bin/fsr4mac"), str(testcase), str(refdir),
         "--fp16" if backend == "f16" else "--i8mpp"], cwd=ROOT, env=env)
    return refdir / "f00_out.bin"


def cmd_install(args):
    log = DIST / "hook.log"
    DIST.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    hook = build_hook(log)
    binary = copy_framework(hook)
    env = os.environ.copy()
    env["FSR4_HOOK_LOG"] = str(log)
    env["FSR4_ROOT"] = str(ROOT)
    env["FSR4_REPLACE"] = "0"
    smoke = build_smoke()
    run(["arch", "-x86_64", str(smoke)], env=env)
    text = log.read_text()
    if "swizzled" not in text or "encodeToCommandBuffer:" not in text:
        raise SystemExit(f"smoke did not swizzle encode:\n{text}")
    dl = build_dlopen()
    log.write_text("")
    run(["arch", "-x86_64", str(dl), str(binary)], env=env)
    loaded = log.read_text()
    if "loaded pid=" not in loaded:
        raise SystemExit(f"copied D3DMetal did not load the hook:\n{loaded}")
    live = build_live()
    env["FSR4_REPLACE"] = "1"
    env["FSR4_SHARPNESS"] = "0"   # the CLI reference frame has no RCAS
    env["FSR4_SETTINGS"] = "none"  # ignore the panel's saved sharpness
    ref = build_reference(env)
    run(["arch", "-x86_64", str(live),
         str(ROOT / "testcases" / "testcase_quality_720to1080"), str(ref)], env=env)
    print(loaded.strip())
    build_panel_app()
    if args.bottle:
        values = {
            KEY: str(binary),
            LOG_KEY: str(DIST / "hook.log"),
            ROOT_KEY: str(ROOT),
            REPLACE_KEY: "1",
        }
        if args.panel:
            values[PANEL_KEY] = "1"   # status/sharpness panel in the game process
        conf_set(Path(args.bottle), values)
    print("original CrossOver.app was not modified")
    print(f"restore: python3 {Path(__file__)} restore")


def cmd_restore(args):
    conf_unset(Path(args.bottle))
    print("CrossOver.app was not modified; next launch uses the bundled D3DMetal")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install")
    i.add_argument("--bottle", default=str(DEFAULT_BOTTLE))
    i.add_argument("--no-bottle", action="store_true")
    i.add_argument("--panel", action="store_true",
                   help="also set FSR4_PANEL=1 in the bottle (status/sharpness panel)")
    i.set_defaults(func=cmd_install)
    r = sub.add_parser("restore")
    r.add_argument("--bottle", default=str(DEFAULT_BOTTLE))
    r.set_defaults(func=cmd_restore)
    args = p.parse_args()
    if args.cmd == "install" and args.no_bottle:
        args.bottle = ""
    args.func(args)


if __name__ == "__main__":
    main()
