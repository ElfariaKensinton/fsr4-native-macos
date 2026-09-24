// Real-domain FP16 equivalents of the int8_NHWC fused operators used by the
// FSR4 model passes. Not from the AMD drop — hand-ported for the Metal port.
//
// Semantics: identical layer graph and math as the int8 ops, except
//   * activations are Tensor3h_NHWC (fp16, real-domain — no requant rounding),
//   * weights are Tensor4h_* (offline-dequantized int8 * tensor scale),
//   * biases are Tensor1h (already fp16 in the original model),
//   * accumulation stays fp32 (matches the i32/int8 accumulation domain),
//   * the int8 saturation clamps are preserved as real-domain clamps:
//     a stage that stored int8 at scale s clamps to [-128*s, 127*s].
//     Relu stages clamp to [0, 127*s].
//
// The float arguments keep the same meaning/order as the int8 op call sites;
// an extra trailing output-scale argument is appended where the int8 code
// took the clamp scale from the output tensor's quantizationScale field.
#pragma once

#include "ml2code_runtime/tensor_float16.hlsli"
#include "ml2code_runtime/globals.hlsli"

// fp32-accumulating dot of two half4s (replicates dot2add pair ordering).
float f16_dot4(half4 a, half4 b, float acc)
{
    acc += float(a.x) * float(b.x);
    acc += float(a.y) * float(b.y);
    acc += float(a.z) * float(b.z);
    acc += float(a.w) * float(b.w);
    return acc;
}

float4 f16_clamp_lo(float4 v, float s) { return clamp(v, -128.0f * s, 127.0f * s); }
float4 f16_clamp_relu(float4 v, float s) { return clamp(v, 0.0f, 127.0f * s); }

// Real-domain requantization: same value the int8 pipeline would hold
// (round to counts at scale s, clamp to the int8 range, back to real).
// Keeps the fp16 path numerically locked to the int8 reference instead of
// drifting on dropped rounding.
float f16_requant(float v, float s) { return clamp(round(v / s), -128.0f, 127.0f) * s; }
float f16_requant_relu(float v, float s) { return clamp(round(v / s), 0.0f, 127.0f) * s; }
float4 f16_requant4(float4 v, float4 s)
{
    return clamp(round(v / s), -128.0f, 127.0f) * s;
}

////////////////////////////////////////////////////////////////////////////
// pass0: 2x2 stride-2 conv, 8ch in -> numFeatures out (Conv_8_16_k2s2b).
// Weights fp16 NHWC, input fp16 NHWC (NHWC_inputs), output fp16.
////////////////////////////////////////////////////////////////////////////
template<typename SI, typename SW, typename SB, typename SO>
void f16_Conv2D_k2s2b(
    const float outputScale,
    const Tensor3h_NHWC<SI> input,
    const Tensor4h_NHWC<SW> weights,
    const Tensor1h<SB> bias,
    const Tensor3h_NHWC<SO> output,
    const ComputeShaderParams computeShaderParams)
{
    const uint3 workPerTick = uint3(1, 1, output.threadGroupSliceSize.z);
    const uint3 perThreadWork = SplitWork(output.threadGroupSliceSize, computeShaderParams.numThreads, workPerTick);
    const int3 workOffset = output.threadGroupSliceStart + computeShaderParams.groupThreadID * perThreadWork;
    const uint numFeatures = weights.logicalSize.w;
    const uint inChannels = weights.logicalSize.z;

    [loop]
    for (uint y = 0; y < perThreadWork.y; y++)
        [loop]
        for (uint x = 0; x < perThreadWork.x; x++)
        {
            const int3 poBase = workOffset + int3(x, y, 0);
            if (!ValidPosition(output, poBase))
                continue;

            const int3 piBase = int3(poBase.xy * 2, 0);

            half4 in_v[2 * 2 * 2];
            [loop]
            for (uint ky = 0; ky < weights.logicalSize.y; ++ky)
                [loop]
                for (uint kx = 0; kx < weights.logicalSize.x; ++kx)
                {
                    const uint base = (ky * weights.logicalSize.x + kx) * (inChannels / 4);
                    [loop]
                    for (uint c = 0; c < inChannels; c += 4)
                        in_v[base + c / 4] = Load4hA(input, piBase + int3(kx, ky, c));
                }

            [loop]
            for (uint f = 0; f < numFeatures; f += 4)
            {
                float4 acc = float4(0, 0, 0, 0);
                [loop]
                for (uint ky = 0; ky < weights.logicalSize.y; ++ky)
                    [loop]
                    for (uint kx = 0; kx < weights.logicalSize.x; ++kx)
                    {
                        const uint base = (ky * weights.logicalSize.x + kx) * (inChannels / 4);
                        [loop]
                        for (uint c = 0; c < inChannels; c += 4)
                        {
                            const half4 v = in_v[base + c / 4];
                            acc.x = f16_dot4(v, Load4hA(weights, int4(kx, ky, c, f + 0)), acc.x);
                            acc.y = f16_dot4(v, Load4hA(weights, int4(kx, ky, c, f + 1)), acc.y);
                            acc.z = f16_dot4(v, Load4hA(weights, int4(kx, ky, c, f + 2)), acc.z);
                            acc.w = f16_dot4(v, Load4hA(weights, int4(kx, ky, c, f + 3)), acc.w);
                        }
                    }
                acc += float4(Load4hA(bias, f));
                Store4hA(output, poBase + int3(0, 0, f),
                         half4(f16_requant4(acc, float4(outputScale, outputScale, outputScale, outputScale))));
            }
        }
}

////////////////////////////////////////////////////////////////////////////
// passes 1/2/12: ConvNextBlock = dw 3x3 conv -> pw expand -> relu -> pw
// contract -> residual. workOffset is per dispatch thread.
////////////////////////////////////////////////////////////////////////////
template<typename SI, typename SW0, typename SB0, typename SW1, typename SB1, typename SW2, typename SB2, typename SO>
void f16_ConvNextBlock(
    const float rcprOutputScale0,
    const float inputScale1,
    const float rcprOutputScale1,
    const float inputScale2,
    const float outputScale,
    const Tensor3h_NHWC<SI> input,
    const Tensor4h_NHWC<SW0> weights0,
    const Tensor1h<SB0> bias0,
    const Tensor4h_NHWC<SW1> weights1,
    const Tensor1h<SB1> bias1,
    const Tensor4h_NHWC<SW2> weights2,
    const Tensor1h<SB2> bias2,
    const Tensor3h_NHWC<SO> output,
    const ComputeShaderParams computeShaderParams)
{
    const int3 poBase = int3(computeShaderParams.dispatchThreadID.xy, 0);
    if (!ValidPosition(output, poBase))
        return;
    const int3 piBase = int3(poBase.xy, 0) - int3(1, 1, 0);
    const uint numFeatures = weights0.logicalSize.w;
    const uint cIn0 = weights0.logicalSize.z;
    const uint cOut1 = weights1.logicalSize.w;
    const uint cIn2 = weights2.logicalSize.z;

    float accumulator[16];
    [loop]
    for (uint i = 0; i < numFeatures; ++i) { accumulator[i] = 0; }

    // First convolution (3x3)
    [loop]
    for (uint ky = 0; ky < weights0.logicalSize.y; ++ky)
        [loop]
        for (uint kx = 0; kx < weights0.logicalSize.x; ++kx)
        {
            if (!ValidPosition(input, int3(piBase.xy + int2(kx, ky), 0)))
                continue;

            half4 vs[16];  // supports cIn0 <= 64
            [loop]
            for (uint c = 0; c < cIn0; c += 4)
                vs[c / 4] = Load4hA(input, piBase + int3(kx, ky, c));

            // Four independent output channels per iteration. Same dot order
            // as the scalar loop, so the requant result stays put.
            [loop]
            for (uint f = 0; f < numFeatures; f += 4)
            {
                float4 acc = float4(accumulator[f], accumulator[f + 1], accumulator[f + 2], accumulator[f + 3]);
                [loop]
                for (uint c = 0; c < cIn0; c += 4)
                {
                    const half4 v = vs[c / 4];
                    acc.x = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 0)), acc.x);
                    acc.y = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 1)), acc.y);
                    acc.z = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 2)), acc.z);
                    acc.w = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 3)), acc.w);
                }
                accumulator[f] = acc.x; accumulator[f + 1] = acc.y;
                accumulator[f + 2] = acc.z; accumulator[f + 3] = acc.w;
            }
        }

    half conv_result[16];
    [loop]
    for (uint f = 0; f < numFeatures; ++f)
    {
        const float v = accumulator[f] + float(Load1hU(bias0, f));
        conv_result[f] = half(f16_requant(v, 1.0f / rcprOutputScale0));
    }

    // Second convolution (1x1 expand) + Relu
    half relu_result[32];
    [loop]
    for (uint i = 0; i < cOut1; i += 4)
    {
        float4 a = float4(Load1hU(bias1, i), Load1hU(bias1, i + 1), Load1hU(bias1, i + 2), Load1hU(bias1, i + 3));
        [loop]
        for (uint c = 0; c < numFeatures; c += 4)
        {
            const half4 conv = half4(conv_result[c + 0], conv_result[c + 1], conv_result[c + 2], conv_result[c + 3]);
            a.x = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 0)), a.x);
            a.y = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 1)), a.y);
            a.z = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 2)), a.z);
            a.w = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 3)), a.w);
        }
        const float s = 1.0f / rcprOutputScale1;
        relu_result[i] = half(f16_requant_relu(a.x, s));
        relu_result[i + 1] = half(f16_requant_relu(a.y, s));
        relu_result[i + 2] = half(f16_requant_relu(a.z, s));
        relu_result[i + 3] = half(f16_requant_relu(a.w, s));
    }

    // Third convolution (1x1 contract) + residual
    const int3 po = int3(poBase.xy, 0);
    [loop]
    for (uint i = 0; i < numFeatures; i += 4)
    {
        float4 r = float4(Load4hA(bias2, i));
        const half4 vin = Load4hA(input, int3(po.xy, i));
        [loop]
        for (uint c = 0; c < cIn2; c += 4)
        {
            const half4 v = half4(relu_result[c + 0], relu_result[c + 1], relu_result[c + 2], relu_result[c + 3]);
            r.x = f16_dot4(v, Load4hA(weights2, int4(0, 0, c, i + 0)), r.x);
            r.y = f16_dot4(v, Load4hA(weights2, int4(0, 0, c, i + 1)), r.y);
            r.z = f16_dot4(v, Load4hA(weights2, int4(0, 0, c, i + 2)), r.z);
            r.w = f16_dot4(v, Load4hA(weights2, int4(0, 0, c, i + 3)), r.w);
        }
        r += float4(vin);
        Store4hA(output, po + int3(0, 0, i),
                 half4(f16_requant4(r, float4(outputScale, outputScale, outputScale, outputScale))));
    }
}

////////////////////////////////////////////////////////////////////////////
// passes 3/6: 2x2 stride-2 conv with split-channel output scales.
////////////////////////////////////////////////////////////////////////////
template<typename SI, typename SW, typename SB, typename SO>
void f16_FusedConv2D_k2s2b_QuantizedOutput(
    const float quantFactor0,
    const float quantFactor1,
    const Tensor3h_NHWC<SI> input,
    const Tensor4h_NHWC<SW> weights,
    const Tensor1h<SB> bias,
    const Tensor3h_NHWC<SO> output,
    const ComputeShaderParams computeShaderParams)
{
    const uint3 workPerTick = uint3(1, 1, output.threadGroupSliceSize.z);
    const uint3 perThreadWork = SplitWork(output.threadGroupSliceSize, computeShaderParams.numThreads, workPerTick);
    const int3 workOffset = output.threadGroupSliceStart + computeShaderParams.groupThreadID * perThreadWork;
    const uint2 convStrides = uint2(2, 2);
    const uint numFeatures = weights.logicalSize.w;
    const uint cIn = weights.logicalSize.z;

    [loop]
    for (uint y = 0; y < perThreadWork.y; y++)
        [loop]
        for (uint x = 0; x < perThreadWork.x; x++)
        {
            const int3 poBase = workOffset + int3(x, y, 0);
            if (!ValidPosition(output, poBase))
                continue;
            const int3 piBase = int3(poBase.xy * convStrides, 0);

            [loop]
            for (uint f = 0; f < numFeatures; f += 4)
            {
                float4 acc = float4(Load4hA(bias, f));
                [loop]
                for (uint ky = 0; ky < weights.logicalSize.y; ++ky)
                    [loop]
                    for (uint kx = 0; kx < weights.logicalSize.x; ++kx)
                    {
                        [loop]
                        for (uint c = 0; c < cIn; c += 4)
                        {
                            const half4 v = Load4hA(input, piBase + int3(kx, ky, c));
                            acc.x = f16_dot4(v, Load4hA(weights, int4(kx, ky, c, f + 0)), acc.x);
                            acc.y = f16_dot4(v, Load4hA(weights, int4(kx, ky, c, f + 1)), acc.y);
                            acc.z = f16_dot4(v, Load4hA(weights, int4(kx, ky, c, f + 2)), acc.z);
                            acc.w = f16_dot4(v, Load4hA(weights, int4(kx, ky, c, f + 3)), acc.w);
                        }
                    }
                const float s0 = (f + 0) < numFeatures / 2 ? quantFactor0 : quantFactor1;
                const float s1 = (f + 1) < numFeatures / 2 ? quantFactor0 : quantFactor1;
                const float s2 = (f + 2) < numFeatures / 2 ? quantFactor0 : quantFactor1;
                const float s3 = (f + 3) < numFeatures / 2 ? quantFactor0 : quantFactor1;
                acc = f16_requant4(acc, float4(s0, s1, s2, s3));
                Store4hA(output, poBase + int3(0, 0, f), half4(acc));
            }
        }
}

////////////////////////////////////////////////////////////////////////////
// FasterNet block core: partial grouped 3x3 conv on the first numFeatures/2
// output channels, concat with input[numFeatures/2:], pw expand x2 + relu,
// pw contract, residual add = input[i] for all i.
// Returns the (clamped) numFeatures vector in fnb[].
////////////////////////////////////////////////////////////////////////////
template<uint numFeatures, uint numGroups, typename SI, typename SW0, typename SB0, typename SW1, typename SB1, typename SW2, typename SB2>
void f16_FNB_core(
    const float quantScale1,
    const float activationScaleFactor,
    const float fnbOutputScale,
    const Tensor3h_NHWC<SI> input,
    const Tensor4h_NHWC<SW0> weights0,
    const Tensor1h<SB0> bias0,
    const Tensor4h_NHWC<SW1> weights1,
    const Tensor1h<SB1> bias1,
    const Tensor4h_NHWC<SW2> weights2,
    const Tensor1h<SB2> bias2,
    const int3 poBase,
    out half fnb[numFeatures])
{
    const int3 piBase = int3(poBase.xy, 0) - int3(1, 1, 0);

    half conv_result[numFeatures];
    // upper half is a passthrough copy of the input
    [loop]
    for (uint z = numFeatures / 2; z < numFeatures; z += 4)
    {
        const half4 v = Load4hA(input, int3(poBase.xy, z));
        conv_result[z + 0] = v.x; conv_result[z + 1] = v.y;
        conv_result[z + 2] = v.z; conv_result[z + 3] = v.w;
    }

    // first (partial, grouped) 3x3 convolution -> numFeatures/2 outputs.
    // group g covers outputs [g*fPerGroup,(g+1)*fPerGroup) reading input
    // channels [g*ciPerGroup,(g+1)*ciPerGroup).
    const uint ciPerGroup = weights0.logicalSize.z;
    const uint fPerGroup = (numFeatures / 2) / numGroups;

    float conv_acc[numFeatures / 2];
    [loop]
    for (uint f = 0; f < numFeatures / 2; ++f)
        conv_acc[f] = float(Load1hU(bias0, f));

    [loop]
    for (uint ky = 0; ky < weights0.logicalSize.y; ++ky)
        [loop]
        for (uint kx = 0; kx < weights0.logicalSize.x; ++kx)
        {
            if (!ValidPosition(input, int3(piBase.xy + int2(kx, ky), 0)))
                continue;

            half4 vs[numFeatures / 4];
            [loop]
            for (uint c = 0; c < ciPerGroup * numGroups; c += 4)
                vs[c / 4] = Load4hA(input, piBase + int3(kx, ky, c));

            [loop]
            for (uint f = 0; f < numFeatures / 2; f += 4)
            {
                const uint g = f / fPerGroup;
                float4 acc = float4(conv_acc[f], conv_acc[f + 1], conv_acc[f + 2], conv_acc[f + 3]);
                [loop]
                for (uint c = 0; c < ciPerGroup; c += 4)
                {
                    const half4 v = vs[g * ciPerGroup / 4 + c / 4];
                    acc.x = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 0)), acc.x);
                    acc.y = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 1)), acc.y);
                    acc.z = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 2)), acc.z);
                    acc.w = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 3)), acc.w);
                }
                conv_acc[f] = acc.x; conv_acc[f + 1] = acc.y;
                conv_acc[f + 2] = acc.z; conv_acc[f + 3] = acc.w;
            }
        }

    [loop]
    for (uint f = 0; f < numFeatures / 2; ++f)
        conv_result[f] = half(f16_requant(conv_acc[f], quantScale1));

    // second convolution: 1x1 expand -> 2*numFeatures + relu
    half relu_result[numFeatures * 2];
    [loop]
    for (uint i = 0; i < numFeatures * 2; i += 4)
    {
        float4 a = float4(Load1hU(bias1, i), Load1hU(bias1, i + 1), Load1hU(bias1, i + 2), Load1hU(bias1, i + 3));
        [loop]
        for (uint c = 0; c < numFeatures; c += 4)
        {
            const half4 conv = half4(conv_result[c + 0], conv_result[c + 1], conv_result[c + 2], conv_result[c + 3]);
            a.x = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 0)), a.x);
            a.y = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 1)), a.y);
            a.z = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 2)), a.z);
            a.w = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 3)), a.w);
        }
        relu_result[i] = half(f16_requant_relu(a.x, activationScaleFactor));
        relu_result[i + 1] = half(f16_requant_relu(a.y, activationScaleFactor));
        relu_result[i + 2] = half(f16_requant_relu(a.z, activationScaleFactor));
        relu_result[i + 3] = half(f16_requant_relu(a.w, activationScaleFactor));
    }

    // third convolution: 1x1 contract -> numFeatures; residual = input[i]
    [loop]
    for (uint i = 0; i < numFeatures; i += 4)
    {
        float4 r = float4(Load4hA(bias2, i));
        [loop]
        for (uint c = 0; c < numFeatures * 2; c += 4)
        {
            const half4 v = half4(relu_result[c + 0], relu_result[c + 1], relu_result[c + 2], relu_result[c + 3]);
            r.x = f16_dot4(v, Load4hA(weights2, int4(0, 0, c, i + 0)), r.x);
            r.y = f16_dot4(v, Load4hA(weights2, int4(0, 0, c, i + 1)), r.y);
            r.z = f16_dot4(v, Load4hA(weights2, int4(0, 0, c, i + 2)), r.z);
            r.w = f16_dot4(v, Load4hA(weights2, int4(0, 0, c, i + 3)), r.w);
        }
        const half4 vin = Load4hA(input, int3(poBase.xy, i));
        r += float4(vin);
        if (fnbOutputScale < 1.0e20f)
            r = f16_requant4(r, float4(fnbOutputScale, fnbOutputScale, fnbOutputScale, fnbOutputScale));
        fnb[i + 0] = half(r.x); fnb[i + 1] = half(r.y);
        fnb[i + 2] = half(r.z); fnb[i + 3] = half(r.w);
    }
}

// plain FasterNetBlock (passes 4/5/7/8/10): two output scales by half.
template<uint numFeatures, uint numGroups, typename SI, typename SW0, typename SB0, typename SW1, typename SB1, typename SW2, typename SB2, typename SO>
void f16_FNB(
    const float quantScale0,
    const float quantScale1,
    const float activationScaleFactor,
    const float quantOutputScale0,
    const float quantOutputScale1,
    const Tensor3h_NHWC<SI> input,
    const Tensor4h_NHWC<SW0> weights0,
    const Tensor1h<SB0> bias0,
    const Tensor4h_NHWC<SW1> weights1,
    const Tensor1h<SB1> bias1,
    const Tensor4h_NHWC<SW2> weights2,
    const Tensor1h<SB2> bias2,
    const Tensor3h_NHWC<SO> output,
    const ComputeShaderParams computeShaderParams)
{
    const uint3 workPerTick = uint3(1, 1, output.threadGroupSliceSize.z);
    const uint3 perThreadWork = SplitWork(output.threadGroupSliceSize, computeShaderParams.numThreads, workPerTick);
    const int3 workOffset = output.threadGroupSliceStart + computeShaderParams.groupThreadID * perThreadWork;
    const int3 poBase = workOffset;

    // The host rounds up to 64 threads. An unchecked x past a tensor row
    // aliases the next row in the linear scratch buffer and races its
    // valid writer (480- and 240-pixel encoder levels at the 1080 tier).
    if (!ValidPosition(output, poBase))
        return;

    half fnb[numFeatures];
    // neutralize the internal clamp; the i8 FNB requantizes to its own
    // intermediate scale (quantScale1 for conv_result, activationScaleFactor
    // for relu, then quantOutputScale* at the store). The store clamp is
    // applied here per channel half instead.
    f16_FNB_core<numFeatures, numGroups>(quantScale1, activationScaleFactor,
        1.0e30f, input, weights0, bias0, weights1, bias1, weights2, bias2,
        poBase, fnb);

    [loop]
    for (uint i = 0; i < numFeatures; i += 4)
    {
        const float s0 = (i + 0) < numFeatures / 2 ? quantOutputScale0 : quantOutputScale1;
        const float s1 = (i + 1) < numFeatures / 2 ? quantOutputScale0 : quantOutputScale1;
        const float s2 = (i + 2) < numFeatures / 2 ? quantOutputScale0 : quantOutputScale1;
        const float s3 = (i + 3) < numFeatures / 2 ? quantOutputScale0 : quantOutputScale1;
        float4 r = float4(fnb[i + 0], fnb[i + 1], fnb[i + 2], fnb[i + 3]);
        r = f16_requant4(r, float4(s0, s1, s2, s3));
        Store4hA(output, poBase + int3(0, 0, i), half4(r));
    }
}

////////////////////////////////////////////////////////////////////////////
// FNB + ConvTranspose2D(2x2 s2) + skip-Add (passes 9/11).
// ct2d weights HWCN: (pk.x,pk.y,oc,ic) — ic contiguous innermost.
// numOutputChannels = output.logicalSize.z (32 for <64,2>, 16 for <32,1>).
////////////////////////////////////////////////////////////////////////////
template<uint numFeatures, uint numGroups, typename SI, typename SA, typename SW0, typename SB0, typename SW1, typename SB1, typename SW2, typename SW3, typename SB2, typename SB3, typename SO>
void f16_FNB_CT2D_ADD(
    const float quantScale0,
    const float quantScale1,
    const float activationScaleFactor,
    const float fnbOutputScale,
    const float quantOutputScale0,
    const float quantOutputScale1,
    const Tensor3h_NHWC<SI> input,
    const Tensor3h_NHWC<SA> inputAdd,
    const Tensor4h_NHWC<SW0> weights0,
    const Tensor1h<SB0> bias0,
    const Tensor4h_NHWC<SW1> weights1,
    const Tensor1h<SB1> bias1,
    const Tensor4h_NHWC<SW2> weights2,
    const Tensor1h<SB2> bias2,
    const Tensor4h_HWCN<SW3> ct2d_weights,
    const Tensor1h<SB3> ct2d_bias,
    const Tensor3h_NHWC<SO> output,
    const ComputeShaderParams computeShaderParams)
{
    const int3 poBase = int3(computeShaderParams.dispatchThreadID.xy, 0);
    if (!ValidPosition(input, poBase))
        return;

    half fnb[numFeatures];
    f16_FNB_core<numFeatures, numGroups>(quantScale1, activationScaleFactor,
        fnbOutputScale, input, weights0, bias0, weights1, bias1, weights2,
        bias2, poBase, fnb);

    const int2 localOffset[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    const uint numOutputChannels = output.logicalSize.z;

    [loop]
    for (int i = 0; i < 4; ++i)
    {
        const int2 pk = localOffset[i];
        const uint2 poBase2D = computeShaderParams.dispatchThreadID.xy * 2 + pk;
        if (any(poBase2D >= output.logicalSize.xy))
            continue;

        [loop]
        for (uint z = 0; z < numOutputChannels; z += 4)
        {
            float4 acc = float4(Load4hA(ct2d_bias, z));
            [loop]
            for (uint oc = 0; oc < 4; ++oc)
            {
                float a = acc[oc];
                [loop]
                for (uint c = 0; c < numFeatures; c += 4)
                {
                    const half4 w = Load4hA(ct2d_weights, int4(pk.x, pk.y, z + oc, c));
                    a += float(fnb[c + 0]) * float(w.x);
                    a += float(fnb[c + 1]) * float(w.y);
                    a += float(fnb[c + 2]) * float(w.z);
                    a += float(fnb[c + 3]) * float(w.w);
                }
                acc[oc] = a;
            }
            acc += float4(Load4hA(inputAdd, int3(poBase2D, z)));
            const float os = (z < numOutputChannels / 2) ? quantOutputScale0 : quantOutputScale1;
            Store4hA(output, int3(poBase2D, z), half4(f16_requant4(acc, float4(os, os, os, os))));
        }
    }
}

////////////////////////////////////////////////////////////////////////////
// post pass: ConvNextBlock producing the 16-channel fnb[] vector (shared by
// all four ConvTranspose2D subpixel taps — the caller hoists it out of the
// tap loop; the int8 original computes it once into packedVss too).
////////////////////////////////////////////////////////////////////////////
template<typename SI, typename SW0, typename SB0, typename SW1, typename SB1, typename SW2, typename SB2>
void f16_CNB(
    const float rcprOutputScale0,
    const float inputScale1,
    const float rcprOutputScale1,
    const float inputScale2,
    const float cnbOutputScale,
    const Tensor3h_NHWC<SI> input,
    const Tensor4h_NHWC<SW0> weights0,
    const Tensor1h<SB0> bias0,
    const Tensor4h_NHWC<SW1> weights1,
    const Tensor1h<SB1> bias1,
    const Tensor4h_NHWC<SW2> weights2,
    const Tensor1h<SB2> bias2,
    const uint2 dispatchThreadID,
    out half fnb[16])
{
    const int3 poBase = int3(dispatchThreadID.xy, 0);
    const int3 piBase = int3(poBase.xy, 0) - int3(1, 1, 0);
    const uint numFeatures = weights0.logicalSize.w;
    const uint cIn0 = weights0.logicalSize.z;

    float accumulator[16];
    [loop]
    for (uint i = 0; i < numFeatures; ++i) { accumulator[i] = 0; }

    [loop]
    for (uint ky = 0; ky < weights0.logicalSize.y; ++ky)
        [loop]
        for (uint kx = 0; kx < weights0.logicalSize.x; ++kx)
        {
            if (!ValidPosition(input, int3(piBase.xy + int2(kx, ky), 0)))
                continue;

            half4 vs[16];  // supports cIn0 <= 64
            [loop]
            for (uint c = 0; c < cIn0; c += 4)
                vs[c / 4] = Load4hA(input, piBase + int3(kx, ky, c));

            [loop]
            for (uint f = 0; f < numFeatures; f += 4)
            {
                float4 acc = float4(accumulator[f], accumulator[f + 1], accumulator[f + 2], accumulator[f + 3]);
                [loop]
                for (uint c = 0; c < cIn0; c += 4)
                {
                    const half4 v = vs[c / 4];
                    acc.x = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 0)), acc.x);
                    acc.y = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 1)), acc.y);
                    acc.z = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 2)), acc.z);
                    acc.w = f16_dot4(v, Load4hA(weights0, int4(kx, ky, c, f + 3)), acc.w);
                }
                accumulator[f] = acc.x; accumulator[f + 1] = acc.y;
                accumulator[f + 2] = acc.z; accumulator[f + 3] = acc.w;
            }
        }

    half conv_result[16];
    [loop]
    for (uint f = 0; f < numFeatures; ++f)
    {
        const float v = accumulator[f] + float(Load1hU(bias0, f));
        conv_result[f] = half(f16_requant(v, 1.0f / rcprOutputScale0));
    }

    half relu_result[32];
    [loop]
    for (uint i = 0; i < weights1.logicalSize.w; i += 4)
    {
        float4 a = float4(Load1hU(bias1, i), Load1hU(bias1, i + 1), Load1hU(bias1, i + 2), Load1hU(bias1, i + 3));
        [loop]
        for (uint c = 0; c < numFeatures; c += 4)
        {
            const half4 conv = half4(conv_result[c + 0], conv_result[c + 1], conv_result[c + 2], conv_result[c + 3]);
            a.x = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 0)), a.x);
            a.y = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 1)), a.y);
            a.z = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 2)), a.z);
            a.w = f16_dot4(conv, Load4hA(weights1, int4(0, 0, c, i + 3)), a.w);
        }
        const float s = 1.0f / rcprOutputScale1;
        relu_result[i] = half(f16_requant_relu(a.x, s));
        relu_result[i + 1] = half(f16_requant_relu(a.y, s));
        relu_result[i + 2] = half(f16_requant_relu(a.z, s));
        relu_result[i + 3] = half(f16_requant_relu(a.w, s));
    }

    const half4 vin0 = Load4hA(input, int3(poBase.xy, 0));
    const half4 vin1 = Load4hA(input, int3(poBase.xy, 4));
    const half4 vin2 = Load4hA(input, int3(poBase.xy, 8));
    const half4 vin3 = Load4hA(input, int3(poBase.xy, 12));
    [loop]
    for (uint i = 0; i < numFeatures; i += 4)
    {
        float4 a = float4(Load1hU(bias2, i), Load1hU(bias2, i + 1), Load1hU(bias2, i + 2), Load1hU(bias2, i + 3));
        [loop]
        for (uint c = 0; c < weights2.logicalSize.z; c += 4)
        {
            const half4 relu = half4(relu_result[c + 0], relu_result[c + 1], relu_result[c + 2], relu_result[c + 3]);
            a.x = f16_dot4(relu, Load4hA(weights2, int4(0, 0, c, i + 0)), a.x);
            a.y = f16_dot4(relu, Load4hA(weights2, int4(0, 0, c, i + 1)), a.y);
            a.z = f16_dot4(relu, Load4hA(weights2, int4(0, 0, c, i + 2)), a.z);
            a.w = f16_dot4(relu, Load4hA(weights2, int4(0, 0, c, i + 3)), a.w);
        }
        const half4 vin = (i == 0) ? vin0 : (i == 4) ? vin1 : (i == 8) ? vin2 : vin3;
        a += float4(vin);
        fnb[i] = half(f16_requant(a.x, cnbOutputScale));
        fnb[i + 1] = half(f16_requant(a.y, cnbOutputScale));
        fnb[i + 2] = half(f16_requant(a.z, cnbOutputScale));
        fnb[i + 3] = half(f16_requant(a.w, cnbOutputScale));
    }
}

// Per-tap ConvTranspose2D: 16 in -> 8 out (4 model params + 4 recurrent).
// ct2d weights HWCN: (pk.x,pk.y,oc,ic) — ic contiguous innermost.
template<typename SW3, typename SB3>
void f16_CT2D_tap(
    const half fnb[16],
    const Tensor4h_HWCN<SW3> ct2d_weights,
    const Tensor1h<SB3> ct2d_bias,
    const int2 pk,
    out float4 modelParams,
    out float4 recurrent)
{
    const uint numCT2Din = ct2d_weights.logicalSize.w;
    const uint numCT2Dout = ct2d_weights.logicalSize.z;

    modelParams = float4(Load4hA(ct2d_bias, 0));
    recurrent = float4(Load4hA(ct2d_bias, 4));
    [loop]
    for (uint oc = 0; oc < numCT2Dout; ++oc)
    {
        float a = 0;
        [loop]
        for (uint c = 0; c < numCT2Din; c += 4)
        {
            const half4 w = Load4hA(ct2d_weights, int4(pk.x, pk.y, oc, c));
            a += float(fnb[c + 0]) * float(w.x);
            a += float(fnb[c + 1]) * float(w.y);
            a += float(fnb[c + 2]) * float(w.z);
            a += float(fnb[c + 3]) * float(w.w);
        }
        if (oc < 4) modelParams[oc] += a; else recurrent[oc - 4] += a;
    }
}

// All four ConvTranspose taps at once. Each tap keeps the scalar dot order;
// the four chains are independent so they issue together.
template<typename SW3, typename SB3>
void f16_CT2D_taps(
    const half fnb[16],
    const Tensor4h_HWCN<SW3> ct2d_weights,
    const Tensor1h<SB3> ct2d_bias,
    out float4 modelParams[4],
    out float4 recurrent[4])
{
    const int2 pk[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    [unroll]
    for (int t = 0; t < 4; ++t)
    {
        modelParams[t] = float4(Load4hA(ct2d_bias, 0));
        recurrent[t] = float4(Load4hA(ct2d_bias, 4));
    }
    const uint numCT2Din = ct2d_weights.logicalSize.w;
    const uint numCT2Dout = ct2d_weights.logicalSize.z;
    [loop]
    for (uint oc = 0; oc < numCT2Dout; ++oc)
    {
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        [loop]
        for (uint c = 0; c < numCT2Din; c += 4)
        {
            const half4 fn = half4(fnb[c + 0], fnb[c + 1], fnb[c + 2], fnb[c + 3]);
            a0 = f16_dot4(fn, Load4hA(ct2d_weights, int4(pk[0].x, pk[0].y, oc, c)), a0);
            a1 = f16_dot4(fn, Load4hA(ct2d_weights, int4(pk[1].x, pk[1].y, oc, c)), a1);
            a2 = f16_dot4(fn, Load4hA(ct2d_weights, int4(pk[2].x, pk[2].y, oc, c)), a2);
            a3 = f16_dot4(fn, Load4hA(ct2d_weights, int4(pk[3].x, pk[3].y, oc, c)), a3);
        }
        if (oc < 4)
        {
            modelParams[0][oc] += a0;
            modelParams[1][oc] += a1;
            modelParams[2][oc] += a2;
            modelParams[3][oc] += a3;
        }
        else
        {
            const uint r = oc - 4;
            recurrent[0][r] += a0;
            recurrent[1][r] += a1;
            recurrent[2][r] += a2;
            recurrent[3][r] += a3;
        }
    }
}
