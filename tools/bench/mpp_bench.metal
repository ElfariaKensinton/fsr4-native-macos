// Microbench: mpp::tensor_ops::convolution2d + matmul2d on the M5 neural
// accelerator at FSR4's real shapes (int8 x int8 -> int32 and fp16).
// Verifies against a CPU reference first, then times.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct ConvP { uint W, H, C, O, TW, TH; };

template <int TW, int TH, int C, int O>
static inline void conv_i8_t(device int8_t *act, device int8_t *wts, device int32_t *dst,
                             constant ConvP &p, uint2 tgid)
{
    auto tA = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        act, dextents<int32_t, 4>((int)C, (int)p.W, (int)p.H, 1));
    auto tW = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        wts, dextents<int32_t, 4>((int)O, (int)C, 3, 3));
    auto tD = tensor<device int32_t, dextents<int32_t, 4>, tensor_inline>(
        dst, dextents<int32_t, 4>((int)O, (int)p.W, (int)p.H, 1));
    constexpr auto d = convolution2d_descriptor(
        int4(O, TW, TH, 1), int4(C, TW + 2, TH + 2, 1), int2(3, 3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    op.set_offsets(int2((int)(tgid.x * TW), (int)(tgid.y * TH)));
    op.run(tA, tW, tD);
}

#define CONV_I8(TW, TH, C, O) \
kernel void conv_i8_##TW##x##TH##_##C##_##O(device int8_t *act [[buffer(0)]], \
    device int8_t *wts [[buffer(1)]], device int32_t *dst [[buffer(2)]], \
    constant ConvP &p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) \
{ conv_i8_t<TW, TH, C, O>(act, wts, dst, p, tgid); }

CONV_I8(32, 4, 16, 16)
CONV_I8(16, 8, 16, 16)
CONV_I8(64, 4, 16, 16)
CONV_I8(32, 8, 16, 16)
CONV_I8(16, 16, 16, 16)
CONV_I8(32, 4, 16, 32)
CONV_I8(32, 4, 32, 32)
CONV_I8(16, 8, 32, 32)

template <int TW, int TH, int C, int O>
static inline void conv_f16_t(device half *act, device half *wts, device float *dst,
                              constant ConvP &p, uint2 tgid)
{
    auto tA = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
        act, dextents<int32_t, 4>((int)C, (int)p.W, (int)p.H, 1));
    auto tW = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
        wts, dextents<int32_t, 4>((int)O, (int)C, 3, 3));
    auto tD = tensor<device float, dextents<int32_t, 4>, tensor_inline>(
        dst, dextents<int32_t, 4>((int)O, (int)p.W, (int)p.H, 1));
    constexpr auto d = convolution2d_descriptor(
        int4(O, TW, TH, 1), int4(C, TW + 2, TH + 2, 1), int2(3, 3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    op.set_offsets(int2((int)(tgid.x * TW), (int)(tgid.y * TH)));
    op.run(tA, tW, tD);
}

#define CONV_F16(TW, TH, C, O) \
kernel void conv_f16_##TW##x##TH##_##C##_##O(device half *act [[buffer(0)]], \
    device half *wts [[buffer(1)]], device float *dst [[buffer(2)]], \
    constant ConvP &p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) \
{ conv_f16_t<TW, TH, C, O>(act, wts, dst, p, tgid); }

CONV_F16(32, 4, 16, 16)
CONV_F16(16, 8, 16, 16)
CONV_F16(32, 8, 16, 16)

// matmul2d int8: per threadgroup, A [M=32][K] from device (pixels x channels),
// B [K][N] weights from device, C int32 [32][N] to device. Loops over M blocks.
struct GemmP { uint M, N, K; };

template <int N, int K>
static inline void gemm_i8_t(device int8_t *A, device int8_t *B, device int32_t *Cm,
                             constant GemmP &p, uint tgid)
{
    auto tA = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        A, dextents<int32_t, 2>((int)K, (int)p.M));
    auto tB = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        B, dextents<int32_t, 2>((int)N, (int)K));
    auto tC = tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>(
        Cm, dextents<int32_t, 2>((int)N, (int)p.M));
    constexpr auto d = matmul2d_descriptor(32, N, K, false, false, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    const int m0 = (int)(tgid * 32);
    auto mA = tA.slice(0, m0);
    auto mC = tC.slice(0, m0);
    op.run(mA, tB, mC);
}

#define GEMM_I8(N, K) \
kernel void gemm_i8_n##N##_k##K(device int8_t *A [[buffer(0)]], \
    device int8_t *B [[buffer(1)]], device int32_t *Cm [[buffer(2)]], \
    constant GemmP &p [[buffer(3)]], uint tgid [[threadgroup_position_in_grid]]) \
{ gemm_i8_t<N, K>(A, B, Cm, p, tgid); }

GEMM_I8(32, 16)
GEMM_I8(16, 32)
GEMM_I8(64, 32)
GEMM_I8(32, 64)
GEMM_I8(128, 64)
GEMM_I8(64, 128)
