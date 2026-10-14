"""Relocate the external POST-CNB cache after the last legacy tensor reader.

No graph tensor pitch, weight, quantizer or floating-point expression changes.
The generated POST only changes its cache address. A separate shader key
and explicit layout metadata prevent mixing the old appended-cache contract.
"""
import re

ARENA_BYTES = {1080: 20880256, 2160: 83232256, 4320: 332352256}
ALIAS_POLICY = "alias_after_pass12_v1"


def post_cnb_layout(tier, alias=False):
    if tier not in ARENA_BYTES:
        raise ValueError("unknown FSR 4.1.1 tensor tier")
    width, height = tier * 8 // 9, tier // 2
    source_bytes = (width + 2) * (height + 2) * 16
    cache_bytes = width * height * 16
    arena = ARENA_BYTES[tier]
    offset = ((source_bytes if alias else arena) + 255) & ~255
    allocated = arena if alias else offset + cache_bytes
    if offset < source_bytes or offset + cache_bytes > allocated:
        raise ValueError("POST-CNB cache overlaps its input or exceeds scratch")
    return {"policy": ALIAS_POLICY if alias else "append_v1", "arena_bytes": arena,
            "source_plane_bytes": source_bytes, "offset": offset, "bytes": cache_bytes,
            "row_stride": width * 16, "allocated_bytes": allocated,
            "valid_after": "post_cnb", "preserves_neural_intermediates_after_post": not alias}


def alias_post_cnb_cache(source, tier):
    """Accept only split POST whose scratch reads are all CNB cache loads."""
    layout = post_cnb_layout(tier, True)
    marker = "// fsr411-generated-post-cnb-alias: tier=%d base=%du\n" % (tier, layout["offset"])
    if source.startswith(marker):
        return source
    if "fsr411-generated-post-cnb-alias:" in source:
        raise ValueError("POST cache already has another alias layout")
    old = post_cnb_layout(tier, False)["offset"]
    before = "ScratchBuffer.Load4(%du +" % old
    expected = 2 if "fsr411-generated-lossy-post:" in source else 1
    accesses = re.findall(r"\bScratchBuffer\.(\w+)\(", source)
    if source.count(before) != expected or accesses != ["Load4"] * expected:
        raise ValueError("POST alias needs an external CNB cache with no other scratch accesses")
    return marker + source.replace(before, "ScratchBuffer.Load4(%du +" % layout["offset"])
