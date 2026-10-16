#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pack the imported FSR 4.1.1 CNB model for native Metal matrix kernels.

Reads the MIT reimplementation's actual weight tables and the verified AMD
4.1.1 initializer. It does not use the older v07 weights or execute upstream
scripts. The twelve neural passes and POST's CNB run natively; PRE, the
thirteen border epilogues and POST's floating filter retain the imported HLSL
path. Rejects an unexpected source shape instead of guessing.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

INITIALIZER_SHA256 = "a54e552ff69a3f7199861417f7ae461d1f67f636ab532c6ec9d8ea5d2963735a"
UP_INITIALIZER_SHA256 = "3c42fc6eabfa9b5b1e489df0e9b2f58e3f9fa22048c584f6839bb27d32a0640e"
U32_MAX = 0xFFFFFFFF
ABI_BYTES = 212
PARAMETER_STRIDE = 256
INPUT_SLOTS = {1: 0, **{i: i for i in range(2, 13)}}
LOSSY_ROUNDING_PASSES = (1, 2, 4, 5, 7, 8, 9, 10, 12)
LOSSY_FOLD_WORDS = {1: 4, 5: 18, 10: 18, 12: 20}
PUBLICATION_LOCK = ".fsr411_native.lock"


def _write_changed_atomic(path, data):
    """Keep a cached file's inode/mtime; publish changed bytes in one rename."""
    try:
        if path.read_bytes() == data:
            return
    except FileNotFoundError:
        pass
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _publish_native(output_dir, manifest, weights, parameters):
    """Serialize writers to one model directory and publish its manifest last.

    Each data file is replaced atomically, so concurrent readers never see
    truncation or a partial write. Metadata describes the exact emitted
    bytes, and unchanged cached models require no file replacements.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(manifest, weights_sha256=hashlib.sha256(weights).hexdigest(), weights_bytes=len(weights),
                    params_sha256=hashlib.sha256(parameters).hexdigest(), params_bytes=len(parameters))
    manifest_path = output_dir / "fsr411_native_manifest.json"
    with (output_dir / PUBLICATION_LOCK).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _write_changed_atomic(output_dir / "fsr411_native_weights.bin", weights)
        _write_changed_atomic(output_dir / "fsr411_native_params.bin", parameters)
        _write_changed_atomic(manifest_path, (json.dumps(metadata, indent=2) + "\n").encode())
    return manifest_path


def require(pattern, text, description, flags=0):
    match = re.search(pattern, text, flags)
    if not match:
        raise ValueError("unexpected 4.1.1 source: " + description)
    return match


def table(text, name, rows):
    body = require(r"static const uint " + name + r"\[" + str(rows) + r"\]\[16\]\s*=\s*\{(.*?)\};",
                   text, name, re.S)[1]
    words = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)u", body)]
    if len(words) != rows * 16:
        raise ValueError(name + ": wrong packed word count")
    return [words[i:i + 16] for i in range(0, len(words), 16)]


def byte(word, channel):
    bits = (word >> (8 * channel)) & 255
    return bits if bits < 128 else bits - 256


def dyadic_exponent(value):
    fraction, exponent = math.frexp(value)
    if fraction != 0.5:
        raise ValueError("native integer tail requires a positive power of two")
    return exponent - 1


def dot_bounds(weights, bias, input_min=-128, input_max=127):
    lo = bias + sum(w * (input_min if w >= 0 else input_max) for w in weights)
    hi = bias + sum(w * (input_max if w >= 0 else input_min) for w in weights)
    return lo, hi


def geometry(tier):
    levels = []
    for level in range(3):
        width, height = (tier * 8 // 9) >> level, (tier // 2) >> level
        row = (width + 2) * 16
        levels.append((width, height, row, row * (height + 2)))
    return levels


def tensor_at(tier, level, base, channels):
    width, height, row, bank = geometry(tier)[level]
    return [base + row + 16, 16, row, width, height, channels, bank if channels > 16 else 0]


def source_identity(path):
    original = path.read_text()
    return original, {"source": path.name, "source_sha256": hashlib.sha256(original.encode()).hexdigest()}


def int_table(text, name, count):
    body = require(r"static const int " + name + r"\[" + str(count) + r"\]\s*=\s*\{(.*?)\};",
                   text, name, re.S)[1]
    values = [int(x.strip()) for x in body.split(",") if x.strip()]
    if len(values) != count or any(x < -(1 << 31) or x >= (1 << 31) for x in values):
        raise ValueError(name + ": wrong int32 literal table")
    return values


def parse_post_cnb(path, tables_path, tier):
    original, result = source_identity(path)
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", original, flags=re.S)
    tables_text = tables_path.read_text()
    w0, w1, w2 = (int_table(tables_text, name, count) for name, count in
                   (("DW_W", 576), ("PW1_W", 128), ("PW2_W", 128)))
    biases = [int_table(tables_text, name, count) for name, count in
              (("DW_S", 16), ("PW1_S", 32), ("PW2_S", 16))]
    conv_function = require(r"conv_result\[f2 / 4\] = \(uint\)pack_clamp_s8\(int4\((\w+)\(acc\[f2\]\)", text, "POST conv requant function")[1]
    expand_function = require(r"relu_result\[i2 / 4\] = \(uint\)pack_clamp_s8\(int4\((\w+)\(r_out\[i2\]\)", text, "POST expansion requant function")[1]
    conv = require(r"int " + conv_function + r"\(int v\)\s*\{ return \(v \+ (\d+) \+ \(\(v >> (\d+)\) & 1\)\) >> \2; \}", text, "POST conv rounding")
    expand = require(r"int " + expand_function + r"\(int v\)\s*\{ v = max\(v, 0\); return \(v \+ (\d+) \+ \(\(v >> (\d+)\) & 1\)\) >> \2; \}", text, "POST expansion rounding")
    for match in (conv, expand):
        if not 0 < int(match[2]) < 31 or int(match[1]) != (1 << (int(match[2]) - 1)) - 1:
            raise ValueError("POST rounding is not a supported nearest-even shift")
    shifts = [int(conv[2]), int(expand[2])]
    require(r"float4 a = float4\([^\n]+\) \* 9\.765625e-04f;", text, "POST accumulator scale")
    require(r"float4 b4 = float4\(unpack_s8s32\(rd\[i4 / 4\]\)\) \* 6\.25e-02f;", text, "POST residual scale")
    require(r"int4 q = int4\(round\(\(b4 \+ a\) \* 16\.0f\)\);\s*packedVss\[i4 / 4\] = \(uint\)pack_clamp_s8\(q\);", text, "POST tail without int16 narrowing")
    width, height, row, _ = geometry(tier)[0]
    require(r"uint4 rd = ScratchBuffer.Load4\(\(uint\)\(\(iy \+ 1\) \* " + str(row) + r" \+ \(ix \+ 1\) \* 16\)\);", text, "POST input pitch")
    c0 = [byte(w0[tap * 64 + oc * 4 + ic // 4], ic % 4)
          for tap in range(9) for ic in range(16) for oc in range(16)]
    c1 = [byte(w1[oc * 4 + ic // 4], ic % 4) for ic in range(16) for oc in range(32)]
    c2 = [byte(w2[oc * 8 + ic // 4], ic % 4) for ic in range(32) for oc in range(16)]
    bounds = []
    for stage, (matrix, channels, count) in enumerate(zip((c0, c1, c2), (16, 32, 16), (144, 16, 32))):
        channel_bounds = [dot_bounds([matrix[k * channels + oc] for k in range(count)], biases[stage][oc],
                                     0 if stage == 2 else -128, 127) for oc in range(channels)]
        low, high = min(x[0] for x in channel_bounds), max(x[1] for x in channel_bounds)
        if stage == 2:
            low -= 128 << 6
            high += 127 << 6
        if max(abs(low), abs(high)) >= (1 << 24):
            raise ValueError("POST accumulator exceeds exact fp32 integer range")
        bounds.append([low, high])
    result.update(kind="cnb", kernel="fsr411_cnb_coop_chain_32x1", tile=[32, 1], threads=32,
                  input=tensor_at(tier, 0, 0, 16), output=[0, 16, width * 16, width, height, 16, 0],
            