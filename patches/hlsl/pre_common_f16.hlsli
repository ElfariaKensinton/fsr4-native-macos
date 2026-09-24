cbuffer cbPass_Weights
{
    uint4 embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight_dwords_C[64];
}

#include "mlsr_optimized_includes.hlsli"

#include "ml2code_runtime/tensor_float16.hlsli"
#include "ml2code_runtime/tensor_int8.hlsli"
#include "ml2code_runtime/fp16_ops.hlsli"

#define RESOLUTION_1080 0
#define RESOLUTION_2160 1
#define RESOLUTION_4320 2

#define GROUP_SIZE_ 16

float FilterWeightCubic(float x, float B, float C)
{
    float y = 0.0f;
    const float x2 = x * x;
    const float x3 = x * x * x;
    if (x < 1)
        y = (12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B);
    else if (x <= 2)
        y = (-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C);

    return y / 6.0f;
}

bool Reproject(in float2 reprojectedPos, in float2 Size, out float3 history_color)
{
    float3 sum = 0.0f;
    float totalWeight = 0.0f;

    [unroll]
    for (int ty = -1; ty <= 2; ++ty)
    {
        for (int tx = -1; tx <= 2; ++tx)
        {
            float2 samplePos = floor(reprojectedPos + float2(tx, ty));

            float3 reprojectedSample = r_history_color[samplePos];
            float2 sampleDist = abs(samplePos - reprojectedPos);
            float filterWeight = FilterWeightCubic(sampleDist.x, 0.0, 0.5) * FilterWeightCubic(sampleDist.y, 0.0, 0.5);

            if (any(samplePos < 0) || any(samplePos >= Size))
                    filterWeight = 0.0f;

            sum += reprojectedSample * filterWeight;
            totalWeight += filterWeight;
        }
    }

    // If total weight is too small, we'll get errors...
    // Just follow the path for off screen reprojection in this case
    if (totalWeight < 0.0001)
        return false;

    history_color = max(sum / totalWeight, 0.0f);
    return true;
}

// The following code is licensed under the MIT license: https://gist.github.com/TheRealMJP/bc503b0b87b643d3505d41eab8b332ae

// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead of 16.
// See http://vec3.ca/bicubic-filtering-in-fewer-taps/ for more details
bool SampleTextureCatmullRom(in float2 reprojectedPos, in float2 texSize, out float3 history_color)
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    float2 samplePos = reprojectedPos * texSize;
    float2 texPos1 = floor(samplePos - 0.5f) + 0.5f;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    float2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    float2 texPos0 = texPos1 - 1;
    float2 texPos3 = texPos1 + 2;
    float2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    float3 result = 0.0f;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos0.x, texPos0.y), 0.0f) * w0.x * w0.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos12.x, texPos0.y), 0.0f) * w12.x * w0.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos3.x, texPos0.y), 0.0f) * w3.x * w0.y;

    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos0.x, texPos12.y), 0.0f) * w0.x * w12.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos12.x, texPos12.y), 0.0f) * w12.x * w12.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos3.x, texPos12.y), 0.0f) * w3.x * w12.y;

    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos0.x, texPos3.y), 0.0f) * w0.x * w3.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos12.x, texPos3.y), 0.0f) * w12.x * w3.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos3.x, texPos3.y), 0.0f) * w3.x * w3.y;

    history_color = result;
    return true;
}

groupshared float myArray[16][16][16];

uint2 Swizzle1DthreadgroupForQuad(uint a, uint xTile, uint yTile)
{
    uint quadNum = a >> 2;
    uint quadIdx = a & 3;
    uint tileStartX = (quadNum % (xTile >> 1)) << 1;
    uint tileStartY = (quadNum / (xTile >> 1)) << 1;
    uint quadOffsetX = quadIdx & 1;
    uint quadOffsetY = quadIdx >> 1;
    return uint2(tileStartX |  quadOffsetX, tileStartY | quadOffsetY);
}


[numthreads(GROUP_SIZE_*GROUP_SIZE_, 1, 1)]
void main(uint3 LocalThreadId : SV_GroupThreadID, uint3 gid : SV_GroupID, uint3 Dtid : SV_DispatchThreadID)
{
    uint2 gtid = Swizzle1DthreadgroupForQuad(LocalThreadId.x, GROUP_SIZE_, GROUP_SIZE_);
    uint2 dtid = gtid + uint2(gid.x * GROUP_SIZE_, gid.y * GROUP_SIZE_);

    if (dtid.x >= width || dtid.y >= height)
        return;

    const float exposureValue = LoadExposureValue();

    const uint dst_idx = dtid.y * width + dtid.x;
    float2 origin_offset = 0.5 * inv_scale - 0.5f;

    float4 upscale_input = optimized_input_upscale_user(dtid, origin_offset, exposureValue);

    const float history_rect_alpha = 0.1f;
    float temporal_sample_weight = upscale_input.a * history_rect_alpha;

    int2 mv_dialated_pos = dilate_mv(dtid, origin_offset, r_depth);
    float2 velocity = LoadInputVelocity(mv_dialated_pos);

    bool disocclusion = local_disocclusion_user(dtid, velocity);

    // on reset frames, model is trained to the current frame as both input and history
    // Otherwise (offscreen reprojection) the default is 0.0f
    float3 history = reset ? upscale_input.rgb : 0.f;

    float4 recurrent;
    // Init recurrent to 0
    recurrent = 0;

    float2 dstSize = { width, height };

    if (disocclusion == 0 && reset == 0)
    {
        ///-----------------------------------------
        /// Reprojection
        ///-----------------------------------------
        const float2 history_xy = float2(dtid)+velocity * dstSize;
        const float2 history_uv = (history_xy + 0.5f) / tex_size;

        bool onscreen = (0 <= history_xy.x && history_xy.x <= (width-1) && 0 <= history_xy.y && history_xy.y <= (height-1));

        if (onscreen)
        {
            //onscreen = Reproject(history_xy, dstSize, history);
            onscreen = SampleTextureCatmullRom(history_uv, tex_size, history);
            {
                recurrent = r_recurrent_0.SampleLevel(g_history_sampler, history_uv, 0);
            }
        }

        // History is unclamped PQ right now. don't saturate at 1.
        history = max(history, 0);
        history = temporal_sample_weight * upscale_input.rgb + (1.0 - temporal_sample_weight) * history;
    }

    rw_reprojected_color[dtid.xy] = DATA_TYPE3(history);

    half4 downscaleInputs[2];

    // Y + Y + Color Distance (cbcr/euclidean)
    {
        float3 input_ycbcr = RGBtoYCbCr(upscale_input.xyz);
        float3 history_ycbcr = RGBtoYCbCr(history);
        const float color_distance = sqrt(pow(input_ycbcr.y - history_ycbcr.y, 2) + pow(input_ycbcr.z - history_ycbcr.z, 2) + 1e-6);

        downscaleInputs[0].x = MODEL_COLOR_NORMALIZE(NETWORK_DATA_TYPE(input_ycbcr.x));
        downscaleInputs[0].y = MODEL_COLOR_NORMALIZE(NETWORK_DATA_TYPE(history_ycbcr.x));
        downscaleInputs[0].z = NETWORK_DATA_TYPE(color_distance);

    }

    // RFM / Recurrent data
    {
        downscaleInputs[0].w = MODEL_RFM_NORMALIZE(NETWORK_DATA_TYPE(recurrent[0]));
        downscaleInputs[1].x = MODEL_RFM_NORMALIZE(NETWORK_DATA_TYPE(recurrent[1]));
        downscaleInputs[1].y = MODEL_RFM_NORMALIZE(NETWORK_DATA_TYPE(recurrent[2]));
        downscaleInputs[1].z = MODEL_RFM_NORMALIZE(NETWORK_DATA_TYPE(recurrent[3]));
        downscaleInputs[1].w = 0;
    }

    // Downscale
    {
        const ConstantBufferStorage<256> storage_embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight = { embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight_dwords };
        const Tensor4h_NHWC< ConstantBufferStorage<256> > weights = {
            uint4(2, 2, 7, 16), // logicalSize
            uint4(0, 0, 0, 0), // threadGroupSliceStart
            uint4(2, 2, 7, 16), // threadGroupSliceSize
            uint4(2, 2, 8, 16), // storageSize
            uint4(16, 32, 2, 64), // storageByteStrides
            uint4(0, 0, 0, 0), // paddingBegin
            uint4(0, 0, 0, 0), // paddingEnd
            0, // threadGroupStorageByteOffset
        storage_embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight };

        const ConstantBufferStorage<8> storage_embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_bias = { embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_bias_dwords };
        const Tensor1h< ConstantBufferStorage<8> > bias = {
            16, // logicalSize
            0, // threadGroupSliceStart
            16, // threadGroupSliceSize
            16, // storageSize
            2, // storageByteStrides
            0, // paddingBegin
            0, // paddingEnd
            0, // threadGroupStorageByteOffset
            storage_embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_bias };

        // quantized_NHWC_/encoder2/ResidualBlock_0/body/input_quantization/act_quant/export_handler/QuantizeLinear_output_0
#if FFX_MLSR_RESOLUTION == RESOLUTION_1080
        const uint3 logicalSize_slice_0 = uint3(960, 540, 16);
        const int3 groupStart_slice_0 = int3(0, 0, 0) + int3(gid.xy, 1) * int3(8, 8, 16);
        const uint3 groupSize_slice_0 = uint3(8, 8, 16);
        const uint3 storageSize_slice_0 = uint3(960, 540, 16);
        const uint3 tensorByteStrides_slice_0 = uint3(16, 15360, 1);
#elif FFX_MLSR_RESOLUTION == RESOLUTION_2160
        const uint3 logicalSize_slice_0 = uint3(1920, 1080, 16);
        const int3 groupStart_slice_0 = int3(0, 0, 0) + int3(gid.xy, 1) * int3(8, 8, 16);
        const uint3 groupSize_slice_0 = uint3(8, 8, 16);
        const uint3 storageSize_slice_0 = uint3(1920, 1080, 16);
        const uint3 tensorByteStrides_slice_0 = uint3(16, 30720, 1);
#elif FFX_MLSR_RESOLUTION == RESOLUTION_4320
        const uint3 logicalSize_slice_0 = uint3(3840, 2160, 16);
        const int3 groupStart_slice_0 = int3(0, 0, 0) + int3(gid.xy, 1) * int3(8, 8, 16);
        const uint3 groupSize_slice_0 = uint3(8, 8, 16);
        const uint3 storageSize_slice_0 = uint3(3840, 2160, 16);
        const uint3 tensorByteStrides_slice_0 = uint3(16, 61440, 1);
#endif
        const int threadGroupByteOffsetInTensor_slice_0 = dot(groupStart_slice_0, tensorByteStrides_slice_0);
        const float quantizationScale_slice_0 = downscale_quantizationScale;
        const RWBufferStorage storage_slice_0 = { ScratchBuffer };
        const uint3 paddingBegin_slice_0 = uint3(0, 0, 0);
        const uint3 paddingEnd_slice_0 = uint3(0, 0, 0);
        const uint3 tensorByteStrides_f16_slice_0 = tensorByteStrides_slice_0 * 2;
        const int threadGroupByteOffsetInTensor_f16_slice_0 = dot(groupStart_slice_0, tensorByteStrides_f16_slice_0);
        const Tensor3h_NHWC<RWBufferStorage> output = { logicalSize_slice_0, groupStart_slice_0, groupSize_slice_0, storageSize_slice_0, tensorByteStrides_f16_slice_0, paddingBegin_slice_0, paddingEnd_slice_0, threadGroupByteOffsetInTensor_f16_slice_0 + 0, storage_slice_0 };

        float accumulators[16];
        [unroll]
        for (uint f = 0 ; f < 16; f++)
        {
            accumulators[f] = 0;
        }

        const float rcpScale = 1.0f / quantizationScale_slice_0;
        uint2 downscaleLocalID = gtid % 2;

        [unroll]
        for (uint f = 0; f < 16; f += 4)
        {
            const uint base = downscaleLocalID.x + downscaleLocalID.y * 2;
            uint4 w0 = embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight_dwords_C[base + (f + 0) * 4];
            uint4 w1 = embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight_dwords_C[base + (f + 1) * 4];
            uint4 w2 = embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight_dwords_C[base + (f + 2) * 4];
            uint4 w3 = embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight_dwords_C[base + (f + 3) * 4];
            half4 h0 = Unpack4h(w0.xy), h1 = Unpack4h(w1.xy), h2 = Unpack4h(w2.xy), h3 = Unpack4h(w3.xy);
            float a0 = dot2add(h0.xy, downscaleInputs[0].xy, accumulators[f]);
            float a1 = dot2add(h1.xy, downscaleInputs[0].xy, accumulators[f + 1]);
            float a2 = dot2add(h2.xy, downscaleInputs[0].xy, accumulators[f + 2]);
            float a3 = dot2add(h3.xy, downscaleInputs[0].xy, accumulators[f + 3]);
            a0 = dot2add(h0.zw, downscaleInputs[0].zw, a0);
            a1 = dot2add(h1.zw, downscaleInputs[0].zw, a1);
            a2 = dot2add(h2.zw, downscaleInputs[0].zw, a2);
            a3 = dot2add(h3.zw, downscaleInputs[0].zw, a3);
            h0 = Unpack4h(w0.zw); h1 = Unpack4h(w1.zw); h2 = Unpack4h(w2.zw); h3 = Unpack4h(w3.zw);
            a0 = dot2add(h0.xy, downscaleInputs[1].xy, a0);
            a1 = dot2add(h1.xy, downscaleInputs[1].xy, a1);
            a2 = dot2add(h2.xy, downscaleInputs[1].xy, a2);
            a3 = dot2add(h3.xy, downscaleInputs[1].xy, a3);
            a0 = dot2add(h0.zw, downscaleInputs[1].zw, a0);
            a1 = dot2add(h1.zw, downscaleInputs[1].zw, a1);
            a2 = dot2add(h2.zw, downscaleInputs[1].zw, a2);
            a3 = dot2add(h3.zw, downscaleInputs[1].zw, a3);
            a0 += QuadReadAcrossX(a0); a0 += QuadReadAcrossY(a0);
            a1 += QuadReadAcrossX(a1); a1 += QuadReadAcrossY(a1);
            a2 += QuadReadAcrossX(a2); a2 += QuadReadAcrossY(a2);
            a3 += QuadReadAcrossX(a3); a3 += QuadReadAcrossY(a3);
            accumulators[f] = a0; accumulators[f + 1] = a1;
            accumulators[f + 2] = a2; accumulators[f + 3] = a3;
        }

        if (all(downscaleLocalID == 0))
        {
            // Add bias + Store (fp16 real-domain: no requant, i8 saturation
            // range preserved as a clamp)
            const uint4 biasDwords = bias.storage.Load4(bias.OffsetOf(0));
            const uint4 biasDwords2 = bias.storage.Load4(bias.OffsetOf(8));

            float4 vs0 = float4(accumulators[0], accumulators[1], accumulators[2], accumulators[3]) + Unpack4h(biasDwords.xy);
            float4 vs1 = float4(accumulators[4], accumulators[5], accumulators[6], accumulators[7]) + Unpack4h(biasDwords.zw);
            float4 vs2 = float4(accumulators[8], accumulators[9], accumulators[10], accumulators[11]) + Unpack4h(biasDwords2.xy);
            float4 vs3 = float4(accumulators[12], accumulators[13], accumulators[14], accumulators[15]) + Unpack4h(biasDwords2.zw);

            const float s = quantizationScale_slice_0;

            int2 localId = int2(gid.xy) * int2(8, 8) + int2(gtid.xy/2);
            const int3 po = int3(localId, 0);
            Store4hA(output, po + int3(0, 0, 0), half4(f16_requant4(vs0, float4(s, s, s, s))));
            Store4hA(output, po + int3(0, 0, 4), half4(f16_requant4(vs1, float4(s, s, s, s))));
            Store4hA(output, po + int3(0, 0, 8), half4(f16_requant4(vs2, float4(s, s, s, s))));
            Store4hA(output, po + int3(0, 0, 12), half4(f16_requant4(vs3, float4(s, s, s, s))));
        }
    }
}
