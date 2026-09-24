// Microbench v2: correct convolution2d usage (dest sliced per tile, source
// dims = whole image). int8->int32 and fp16->fp32, device and threadgroup dest.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct ConvP { uint W, H, C, O, TW, TH; };

// destination in device memory (int32)
template <int TW, int TH, int C, int O, int W, int H>
static inline void conv_i8_dv(device int8_t *act, device int8_t *wts, device int32_t *dst,
                              uint2 tgid)
{
    auto tA = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        act, dextents<int32_t, 4>(C, W, H, 1));
    auto tW = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        wts, dextents<int32_t, 4>(O, C, 3, 3));
    auto tD = tensor<device int32_t, dextents<int32_t, 4>, tensor_inline>(
        dst, dextents<int32_t, 4>(O, W, H, 1));
    constexpr auto d = convolution2d_descriptor(
        int4(O, TW, TH, 1), int4(C, W, H, 1), int2(3, 3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    auto sD = tD.slice(0, (int)(tgid.x * TW), (int)(tgid.y * TH), 0);
    op.run(tA, tW, sD);
}

// destination in threadgroup memory (int32), then a cheap reduction store so
// the work is not dead.
template <int TW, int TH, int C, int O, int W, int H>
static inline void conv_i8_tg(device int8_t *act, device int8_t *wts, device int32_t *dst,
                              threadgroup int32_t *tile, uint2 tgid, uint tid)
{
    auto tA = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        act, dextents<int32_t, 4>(C, W, H, 1));
    auto tW = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        wts, dextents<int32_t, 4>(O, C, 3, 3));
    auto tD = tensor<threadgroup int32_t, dextents<int32_t, 4>, tensor_inline>(
        tile, dextents<int32_t, 4>(O, TW, TH, 1));
    constexpr auto d = convolution2d_descriptor(
        int4(O, TW, TH, 1), int4(C, W, H, 1), int2(3, 3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    op.set_offsets(int2((int)(tgid.x * TW), (int)(tgid.y * TH)));
    op.run(tA, tW, tD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // each thread writes one int8-requantized value per 4 outputs: emulate the
    // real epilogue cost roughly (requant + store int8-ish)
    const uint n = TW * TH * O;
    for (uint i = tid; i < n; i += 128) {
        int v = tile[i];
        dst[(tgid.y * ((W + TW - 1) / TW) + tgid.x) * n + i] = v;
    }
}

#define CONV_DV(TW, TH, C, O, W, H) \
kernel void conv_dv_##TW##x##TH##_##C##_##O##_##W##x##H(device int8_t *act [[buffer(0)]], \
    device int8_t *wts [[buffer(1)]], device int32_t *dst [[buffer(2)]], \
    constant ConvP &p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) \
{ conv_i8_dv<TW, TH, C, O, W, H>(act, wts, dst, tgid); }

#define CONV_TG(TW, TH, C, O, W, H) \
kernel void conv_tg_##TW##x##TH##_##C##_##O##_##W##x##H(device int8_t *act [[buffer(0)]], \
    device int8_t *wts [[buffer(1)]], device int32_t *dst [[buffer(2)]], \
    constant ConvP &p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) \
{ threadgroup int32_t tile[TW * TH * O]; conv_i8_tg<TW, TH, C, O, W, H>(act, wts, dst, tile, tgid, tid); }

// small correctness (96x20) incl. borders; plus bigger-source-dims variant
CONV_DV(32, 4, 16, 16, 96, 20)
CONV_DV(32, 4, 16, 16, 128, 32)     // source dims larger than the real image (96x20 tensor)
CONV_TG(32, 4, 16, 16, 96, 20)
// enc0 960x540
CONV_DV(32, 4, 16, 16, 960, 540)
CONV_DV(64, 4, 16, 16, 960, 540)
CONV_DV(32, 8, 16, 16, 960, 540)
CONV_DV(16, 8, 16, 16, 960, 540)
CONV_DV(64, 8, 16, 16, 960, 540)
CONV_TG(32, 4, 16, 16, 960, 540)
CONV_TG(64, 4, 16, 16, 960, 540)
CONV_TG(32, 8, 16, 16, 960, 540)
CONV_TG(64, 8, 16, 16, 960, 540)
// enc1 480x270 (FNB32 dw: C16->O16), enc2 240x135 (FNB64 dw group: C16->O32)
CONV_TG(32, 4, 16, 16, 480, 270)
CONV_TG(64, 4, 16, 16, 480, 270)
CONV_TG(32, 4, 16, 32, 240, 135)
CONV_TG(16, 8, 16, 32, 240, 135)

// fp16 device dest, enc0
template <int TW, int TH, int C, int O, int W, int H>
static inline void conv_f16_dv(device half *act, device half *wts, device float *dst, uint2 tgid)
{
    auto tA = tensor<device half, dextents<int32_t, 4>, tensor_inline>(act, dextents<int32_t, 4>(C, W, H, 1));
    auto tW = tensor<device half, dextents<int32_t, 4>, tensor_inline>(wts, dextents<int32_t, 4>(O, C, 3, 3));
    auto tD = tensor<device float, dextents<int32_t, 4>, tensor_inline>(dst, dextents<int32_t, 4>(O, W, H, 1));
    constexpr auto d = convolution2d_descriptor(
        int4(O, TW, TH, 1), int4(C, W, H, 1), int2(3, 3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    auto sD = tD.slice(0, (int)(tgid.x * TW), (int)(tgid.y * TH), 0);
    op.run(tA, tW, sD);
}
#define CONV_F16(TW, TH, C, O, W, H) \
kernel void convf_dv_##TW##x##TH##_##C##_##O##_##W##x##H(device half *act [[buffer(0)]], \
    device half *wts [[buffer(1)]], device float *dst [[buffer(2)]], \
    constant ConvP &p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) \
{ conv_f16_dv<TW, TH, C, O, W, H>(act, wts, dst, tgid); }
CONV_F16(32, 4, 16, 16, 96, 20)
CONV_F16(32, 4, 16, 16, 960, 540)
CONV_F16(64, 4, 16, 16, 960, 540)

// matmul2d int8 compute-bound probe: A tile [32][K] staged to threadgroup,
// B [K][N] device weights, C [32][N] int32 threadgroup; REP runs per tile.
struct GemmP { uint M, N, K, rep; };
template <int N, int K>
static inline void gemm_tg_t(device int8_t *A, device int8_t *B, device int32_t *Cm,
                             constant GemmP &p, threadgroup int8_t *As, threadgroup int32_t *Cs,
                             uint tgid, uint tid)
{
    const uint m0 = tgid * 32;
    for (uint i = tid; i < 32 * K; i += 128) As[i] = A[m0 * K + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tA = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(As, dextents<int32_t, 2>(K, 32));
    auto tB = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(N, K));
    auto tC = tensor<threadgroup int32_t, dextents<int32_t, 2>, tensor_inline>(Cs, dextents<int32_t, 2>(N, 32));
    constexpr auto d = matmul2d_descriptor(32, N, K, false, false, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    for (uint r = 0; r < p.rep; r++) {
        op.run(tA, tB, tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (r + 1 < p.rep) { As[tid] ^= (int8_t)(Cs[tid] & 1); threadgroup_barrier(mem_flags::mem_threadgroup); }
    }
    for (uint i = tid; i < 32 * N; i += 128) Cm[m0 * N + i] = Cs[i];
}
#define GEMM_TG(N, K) \
kernel void gemm_tg_n##N##_k##K(device int8_t *A [[buffer(0)]], device int8_t *B [[buffer(1)]], \
    device int32_t *Cm [[buffer(2)]], constant GemmP &p [[buffer(3)]], \
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) \
{ threadgroup int8_t As[32 * K]; threadgroup int32_t Cs[32 * N]; gemm_tg_t<N, K>(A, B, Cm, p, As, Cs, tgid, tid); }
GEMM_TG(32, 16)
GEMM_TG(16, 32)
GEMM_TG(64, 32)
GEMM_TG(32, 64)
GEMM_TG(128, 64)
GEMM_TG(64, 128)

// (a) source dims larger than the real image, with offsets; (b) 2x2 stride-2
// conv (passes 3/6); (c) grouped conv via a channel-sliced strided view.
template <int TW, int TH, int C, int O, int W, int H>
static inline void conv_big_tg(device int8_t *act, device int8_t *wts, device int32_t *dst,
                               threadgroup int32_t *tile, constant ConvP &p, uint2 tgid, uint tid)
{
    auto tA = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        act, dextents<int32_t, 4>(C, (int)p.W, (int)p.H, 1));     // runtime extents smaller than W,H
    auto tW = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(wts, dextents<int32_t, 4>(O, C, 3, 3));
    auto tD = tensor<threadgroup int32_t, dextents<int32_t, 4>, tensor_inline>(tile, dextents<int32_t, 4>(O, TW, TH, 1));
    constexpr auto d = convolution2d_descriptor(int4(O, TW, TH, 1), int4(C, W, H, 1), int2(3, 3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    op.set_offsets(int2((int)(tgid.x * TW), (int)(tgid.y * TH)));
    op.run(tA, tW, tD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = TW * TH * O;
    const uint gx = (p.W + TW - 1) / TW;
    for (uint i = tid; i < n; i += 128) dst[(tgid.y * gx + tgid.x) * n + i] = tile[i];
}
kernel void conv_big_32x4_16_16_128x32(device int8_t *act [[buffer(0)]], device int8_t *wts [[buffer(1)]],
    device int32_t *dst [[buffer(2)]], constant ConvP &p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{ threadgroup int32_t tile[32 * 4 * 16]; conv_big_tg<32, 4, 16, 16, 128, 32>(act, wts, dst, tile, p, tgid, tid); }

// 2x2 stride 2: source W,H -> dest W/2,H/2
template <int TW, int TH, int C, int O, int W, int H>
static inline void conv_s2_tg(device int8_t *act, device int8_t *wts, device int32_t *dst,
                              threadgroup int32_t *tile, constant ConvP &p, uint2 tgid, uint tid)
{
    auto tA = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(act, dextents<int32_t, 4>(C, W, H, 1));
    auto tW = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(wts, dextents<int32_t, 4>(O, C, 2, 2));
    auto tD = tensor<threadgroup int32_t, dextents<int32_t, 4>, tensor_inline>(tile, dextents<int32_t, 4>(O, TW, TH, 1));
    constexpr auto d = convolution2d_descriptor(int4(O, TW, TH, 1), int4(C, W, H, 1), int2(2, 2),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(2, 2), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    op.set_offsets(int2((int)(tgid.x * TW * p.C), (int)(tgid.y * TH * p.C)));   // p.C abused as offset multiplier
    op.run(tA, tW, tD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = TW * TH * O;
    const uint gx = (W / 2 + TW - 1) / TW;
    for (uint i = tid; i < n; i += 128) dst[(tgid.y * gx + tgid.x) * n + i] = tile[i];
}
kernel void conv_s2_32x4_16_32_96x20(device int8_t *act [[buffer(0)]], device int8_t *wts [[buffer(1)]],
    device int32_t *dst [[buffer(2)]], constant ConvP &p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{ threadgroup int32_t tile[32 * 4 * 32]; conv_s2_tg<32, 4, 16, 32, 96, 20>(act, wts, dst, tile, p, tgid, tid); }
kernel void conv_s2_32x4_16_32_960x540(device int8_t *act [[buffer(0)]], device int8_t *wts [[buffer(1)]],
    device int32_t *dst [[buffer(2)]], constant ConvP &p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{ threadgroup int32_t tile[32 * 4 * 32]; conv_s2_tg<32, 4, 16, 32, 960, 540>(act, wts, dst, tile, p, tgid, tid); }

// grouped: activation has CT channels, conv reads channels [g*C, (g+1)*C) via strided view
template <int TW, int TH, int C, int CT, int O, int W, int H>
static inline void conv_grp_tg(device int8_t *act, device int8_t *wts, device int32_t *dst,
                               threadgroup int32_t *tile, constant ConvP &p, uint2 tgid, uint tid)
{
    const int g = (int)p.TH;   // abuse: group index passed in p.TH
    auto tA = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        act + g * C, dextents<int32_t, 4>(C, W, H, 1), array<int, 4>{1, CT, CT * W, CT * W * H});
    auto tW = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(wts, dextents<int32_t, 4>(O, C, 3, 3));
    auto tD = tensor<threadgroup int32_t, dextents<int32_t, 4>, tensor_inline>(tile, dextents<int32_t, 4>(O, TW, TH, 1));
    constexpr auto d = convolution2d_descriptor(int4(O, TW, TH, 1), int4(C, W, H, 1), int2(3, 3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    op.set_offsets(int2((int)(tgid.x * TW), (int)(tgid.y * TH)));
    op.run(tA, tW, tD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = TW * TH * O;
    const uint gx = (W + TW - 1) / TW;
    for (uint i = tid; i < n; i += 128) dst[(tgid.y * gx + tgid.x) * n + i] = tile[i];
}
kernel void conv_grp_32x4_16of32_16_96x20(device int8_t *act [[buffer(0)]], device int8_t *wts [[buffer(1)]],
    device int32_t *dst [[buffer(2)]], constant ConvP &p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{ threadgroup int32_t tile[32 * 4 * 16]; conv_grp_tg<32, 4, 16, 32, 16, 96, 20>(act, wts, dst, tile, p, tgid, tid); }
