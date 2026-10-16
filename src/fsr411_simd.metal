// SPDX-License-Identifier: MIT
// Apple7+ (M1–M4) FSR 4.1.1 integer ConvNext blocks on simdgroup matrices.
//
// INT8 activations and weights are exact in half. 8x8 half x half products
// accumulate in float: every model accumulator is bounded by
// 128 * sum|w| < 2^24 (474,880 for the pinned Normal/UP, Exact/Lossy models),
// so the float sum is the exact integer in any association order. Biases,
// residuals and the INT32 epilogue (fsr411::finish) are applied in integers,
// so the result is byte-identical to the scalar backend.
//
// The epilogue addresses the two elements each lane owns in an 8x8 matrix.
// That ownership is the Apple7+ layout used by lane_origin(); the host runs
// fsr411_simd_probe once per device and selects this backend only if both
// the layout and the exact float accumulation are confirmed.
//
// Bindings: 0 scratch, 1 packed INT8 weights + INT32 biases, 2 PassParams,
// 3 the same weights blob widened element-wise to half (same indices).
// Compile with -std=metal3.0 -I src; FSR411_SPECIALIZE=1 for per-pass
// function constants.
#include <metal_stdlib>
#include "fsr411_model.h"

using namespace metal;

namespace fsr411_simd {

typedef simdgroup_matrix<half, 8, 8> hmat;
typedef simdgroup_matrix<float, 8, 8> fmat;

// (column, row) of the first of two adjacent elements owned by this lane.
__attribute__((always_inline)) inline ushort2 lane_origin(ushort lane) {
    const ushort q = lane / 4u;
    return ushort2((q & 2u) * 2u + (lane % 2u) * 2u, (q & 4u) + (lane / 2u) % 4u);
}

__attribute__((always_inline)) inline int2 load_bias2(device const uchar *weights, uint offset, uint channel) {
    return offset == fsr411::kNoOffset ? int2(0) : *(device const int2 *)(weights + offset + channel * 4u);
}

__attribute__((always_inline)) inline ushort pack2(int a, int b) {
    return ushort(uchar(a & 255)) | ushort(uchar(b & 255) << 8);
}

__attribute__((always_inline)) inline fmat zero() {
    return make_filled_simdgroup_matrix<float, 8, 8>(0.f);
}

// Requantize one accumulator fragment into an INT8-valued half fragment.
__attribute__((always_inline)) inline hmat requant(thread fmat &acc, int2 bias, fsr411::Epilogue q) {
    hmat r;
    r.thread_elements()[0] = half(fsr411::finish(int(acc.thread_elements()[0]), bias.x, 0, q));
    r.thread_elements()[1] = half(fsr411::finish(int(acc.thread_elements()[1]), bias.y, 0, q));
    return r;
}

// Copy a W x H pixel window of C channels (multiple of 16) as half, zero
// outside the logical tensor. Layout [row][column][channel].
template <uint C, uint W, uint H, uint THREADS>
__attribute__((always_inline)) inline void load_window(threadgroup half *tile, device const uchar *scratch,
                                                       fsr411::Tensor t, int x0, int y0, uint tid) {
    constexpr uint UNITS = C / 16u;
    for (uint i = tid; i < W * H * UNITS; i += THREADS) {
        const uint px = i / UNITS, ch = (i % UNITS) * 16u;
        const int x = x0 + int(px % W), y = y0 + int(px / W);
        uint4 v(0);
        if (x >= 0 && y >= 0 && uint(x) < t.width && uint(y) < t.height)
            v = *(device const uint4 *)(scratch + fsr411::tensor_address(t, uint(x), uint(y), ch));
        threadgroup half4 *dst = (threadgroup half4 *)(tile + px * C + ch);
        dst[0] = half4(as_type<char4>(v.x));
        dst[1] = half4(as_type<char4>(v.y));
        dst[2] = half4(as_type<char4>(v.z));
        dst[3] = half4(as_type<char4>(v.w));
    }
}

// Tile-local (column, row) of A-matrix row r in pixel block m. Direct: block m
// is 8 consecutive pixels of row sg. Winograd: row r is 2x2 tile r of the
// SIMD group's two rows and block m = 2*i + j its sub-position (j, i).
template <bool WINO>
__attribute__((always_inline)) inline uint2 pixel(uint m, uint r, uint sg) {
    return WINO ? uint2(2u * r + (m & 1u), 2u * sg + (m >> 1u)) : uint2(m * 8u + r, sg);
}

template <bool WINO> struct Conv3x3;

// Direct grouped 3x3 convolution (16 -> 16 per bank): 9 taps x 2 K blocks.
template <> struct Conv3x3<false> {
    template <uint NF, uint BANKS, uint MB, uint SG>
    static __attribute__((always_inline)) inline void run(thread hmat (&a1)[MB][NF / 8u], threadgroup const half *tile,
                                                          device const half *wh, device const uchar *weights,
                                                          FSR411_PARAMS_ADDRESS fsr411::PassParams &p, ushort sg, ushort2 lo) {
        constexpr uint WW = 8u * MB + 2u;
#pragma clang loop unroll(full)
        for (uint b = 0; b < BANKS; ++b) {
            fmat acc[MB][2];
#pragma clang loop unroll(full)
            for (uint m = 0; m < MB; ++m) acc[m][0] = acc[m][1] = zero();
            device const half *w0 = wh + p.weights[0] + b * 2304u;
#pragma clang loop unroll(full)
            for (uint tap = 0; tap < 9u; ++tap) {
                const uint ky = tap / 3u, kx = tap % 3u;
#pragma clang loop unroll(full)
                for (uint kb = 0; kb < 2u; ++kb) {
                    hmat lower, upper;
                    simdgroup_load(lower, w0 + (tap * 16u + kb * 8u) * 16u, 16);
                    simdgroup_load(upper, w0 + (tap * 16u + kb * 8u) * 16u + 8u, 16);
#pragma clang loop unroll(full)
                    for (uint m = 0; m < MB; ++m) {
                        hmat a;
                        simdgroup_load(a, tile + ((sg + ky) * WW + m * 8u + kx) * NF + b * 16u + kb * 8u, NF);
                        simdgroup_multiply_accumulate(acc[m][0], a, lower, acc[m][0]);
                        simdgroup_multiply_accumulate(acc[m][1], a, upper, acc[m][1]);
                    }
                }
            }
#pragma clang loop unroll(full)
            for (uint n = 0; n < 2u; ++n) {
                const int2 bias = load_bias2(weights, p.biases[0], b * 16u + n * 8u + lo.x);
#pragma clang loop unroll(full)
                for (uint m = 0; m < MB; ++m) a1[m][2u * b + n] = requant(acc[m][n], bias, p.q[0]);
            }
        }
    }
};

// Integer Winograd F(2x2, 3x3): 16 products per 2x2 output instead of 36.
// d' = B^T d B and w' = (2G) g (2G)^T are integers (|d'| <= 512, |w'| <= 2048
// checked by the generator), exact in half; y = A^T (d' . w') A = 4 * conv
// exactly while every partial sum stays below 2^24 (generator bound, host
// probe). Weights are [bank][position 4*xi+nu][input 16][output 16].
template <> struct Conv3x3<true> {
    static constexpr int at(uint i, uint k) { return i == 0u ? (k < 3u ? 1 : 0) : (k == 0u ? 0 : k == 1u ? 1 : -1); }
    template <uint NF, uint BANKS, uint MB, uint SG>
    static __attribute__((always_inline)) inline void run(thread hmat (&a1)[MB][NF / 8u], threadgroup const half *tile,
                                                          device const half *wh, device const uchar *weights,
                                                          FSR411_PARAMS_ADDRESS fsr411::PassParams &p, ushort sg, ushort2 lo) {
        static_assert(MB == 4u, "Winograd blocks are the four 2x2 sub-positions");
        constexpr uint WW = 18u;
#pragma clang loop unroll(full)
        for (uint b = 0; b < BANKS; ++b) {
            // This lane's 4x4 input patch of 2x2 tile lo.y, channels lo.x, lo.x+1.
            threadgroup const half *base = tile + (2u * sg * WW + 2u * lo.y) * NF + b * 16u + lo.x;
            half2 d[2][4][4];
#pragma clang loop unroll(full)
            for (uint kb = 0; kb < 2u; ++kb)
#pragma clang loop unroll(full)
                for (uint r = 0; r < 4u; ++r)
#pragma clang loop unroll(full)
                    for (uint c = 0; c < 4u; ++c)
                        d[kb][r][c] = *(threadgroup const half2 *)(base + (r * WW + c) * NF + kb * 8u);
            fmat y[4][2];
#pragma clang loop unroll(full)
            for (uint m = 0; m < 4u; ++m) y[m][0] = y[m][1] = zero();
            device const half *w0 = wh + p.weights[0] + b * 4096u;
#pragma clang loop unroll(full)
            for (uint xi = 0; xi < 4u; ++xi) {
                half2 e[2][4];  // rows of B^T d
#pragma c