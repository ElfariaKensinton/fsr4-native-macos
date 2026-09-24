#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct DP { uint W, H; int ox, oy, sx, sy, dx, dy; uint useSliceA, useSliceD; };

template <int DW, int DH, int SW, int SH>
static inline void diag_t(device int8_t *act, device int8_t *wts, device int32_t *dst,
                          constant DP &p, uint2 tgid)
{
    auto tA = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        act, dextents<int32_t, 4>(16, (int)p.W, (int)p.H, 1));
    auto tW = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        wts, dextents<int32_t, 4>(16, 16, 3, 3));
    auto tD = tensor<device int32_t, dextents<int32_t, 4>, tensor_inline>(
        dst, dextents<int32_t, 4>(16, (int)p.W, (int)p.H, 1));
    constexpr auto d = convolution2d_descriptor(
        int4(16, DW, DH, 1), int4(16, SW, SH, 1), int2(3, 3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    int tx = (int)tgid.x * DW, ty = (int)tgid.y * DH;
    op.set_offsets(int2(p.ox + (p.ox >= 0 ? tx : 0), p.oy + (p.oy >= 0 ? ty : 0)));
    if (p.useSliceA && p.useSliceD) {
        auto sA = tA.slice(0, tx + p.sx, ty + p.sy, 0);
        auto sD = tD.slice(0, tx + p.dx, ty + p.dy, 0);
        op.run(sA, tW, sD);
    } else if (p.useSliceA) {
        auto sA = tA.slice(0, tx + p.sx, ty + p.sy, 0);
        op.run(sA, tW, tD);
    } else if (p.useSliceD) {
        auto sD = tD.slice(0, tx + p.dx, ty + p.dy, 0);
        op.run(tA, tW, sD);
    } else {
        op.run(tA, tW, tD);
    }
}

#define DIAG(DW, DH, SW, SH) \
kernel void diag_##DW##x##DH##_##SW##x##SH(device int8_t *act [[buffer(0)]], \
    device int8_t *wts [[buffer(1)]], device int32_t *dst [[buffer(2)]], \
    constant DP &p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) \
{ diag_t<DW, DH, SW, SH>(act, wts, dst, p, tgid); }

DIAG(32, 4, 34, 6)
DIAG(96, 20, 96, 20)
DIAG(32, 4, 96, 20)
DIAG(32, 4, 32, 4)
DIAG(96, 20, 98, 22)
