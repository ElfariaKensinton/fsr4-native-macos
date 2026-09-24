#include "mlsr_optimized_includes.hlsli"

#define RESOLUTION_1080 0
#define RESOLUTION_2160 1
#define RESOLUTION_4320 2

float3 apply_model_filter(uint x, uint y, const float4 filter, const float exposureValue)
{
    const uint kernelSize = 3;

    const float correlation_factor = ((exp(filter[0]) - exp(-filter[0])) / (exp(filter[0]) + exp(-filter[0])));
    const float scale_factor_x = 2.0 / (1.0 + exp(-filter[1]));
    const float scale_factor_y = 2.0 / (1.0 + exp(-filter[2]));

    const float scale_factor_x2 = scale_factor_x * scale_factor_x;
    const float scale_factor_y2 = scale_factor_y * scale_factor_y;
    const float scale_factor_xy = scale_factor_x * scale_factor_y;

    const float2 lr_pos_f = (-0.5 + inv_scale / 2 + float2(x, y) * inv_scale) + jitter;
    int2 lr_pos_i = round(lr_pos_f);
    if (kernelSize % 2 == 0)
        lr_pos_i = floor(lr_pos_f);

    const uint kernel_radius = kernelSize >> 1;
    float3 upscaled_color = 0.f;
    float total_weight = 0.f;
    [unroll]
    for (uint dy = 0; dy < kernelSize; ++dy)
    {
        const uint y_in = lr_pos_i.y + dy - kernel_radius;
        const float y_dist = ((float)y_in - lr_pos_f.y) * scale.y;
        const float y_dist2 = y_dist * y_dist;

        [unroll]
        for (uint dx = 0; dx < kernelSize; ++dx)
        {
            const uint x_in = lr_pos_i.x + dx - kernel_radius;
            const float x_dist = ((float)x_in - lr_pos_f.x) * scale.x;
            const float x_dist2 = x_dist * x_dist;

            const float gauss_weight = exp((-0.5 / (0.47 * 0.47)) * (x_dist2 * scale_factor_x2 + 2 * correlation_factor * (x_dist * y_dist) * scale_factor_xy + y_dist2 * scale_factor_y2));
            total_weight += gauss_weight;

            upscaled_color += ApplyMuLaw(LoadInputColor(uint2(x_in, y_in)) * exposureValue) * gauss_weight;
        }
    }
    upscaled_color /= total_weight;

    const float blending_factor = (1.0 / (1.0 + exp(-filter[3])));

    float3 res = upscaled_color * (1.0 - blending_factor) + r_reprojected_color[uint2(x, y)] * blending_factor;
    return res;
}

#include "ml2code_runtime/tensor_float16.hlsli"
#include "ml2code_runtime/scalar_functions.hlsli"
#include "ml2code_runtime/fp16_ops.hlsli"

// FP16 real-domain port of the fused CNB + ConvTranspose2D post op.
// Input is the fp16 scratch tensor; weights are Tensor4h (offline-dequantized
// int8*scale); int8 saturation bounds are preserved as real-domain clamps.
template<typename SI, typename SW0, typename SB0, typename SW1, typename SB1, typename SW2, typename SW3, typename SB2, typename SB3, typename SO>
void FusedConv2D_DW_Conv2D_PW_Relu_Conv2D_Add_ConvTranspose2D(
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
    const Tensor4h_HWCN<SW3> ct2d_weights,
    const Tensor1h<SB3> ct2d_bias,
    const Tensor3h_NHWC<SO> output,
    const uint3 numThreads,
    const uint3 groupThreadID,
    const uint3 groupID,
    const uint3 dispatchThreadID)
{
    // The host rounds to 16x16 output tiles. This thread owns a 2x2 block;
    // skip blocks outside the frame before reading the model tensor.
    if (any(dispatchThreadID.xy * 2 >= uint2(width, height)))
        return;
    const int2 localOffset[4] = { int2(0,0), int2(1,0), int2(0,1), int2(1,1) };

    const float exposureValue = LoadExposureValue();
    const bool shouldConvertColorspace = !rcas_enabled;

    // ConvNextBlock once — shared by all four ConvTranspose2D subpixel taps.
    half fnb[16];
    f16_CNB(rcprOutputScale0, inputScale1, rcprOutputScale1,
            inputScale2, cnbOutputScale, input,
            weights0, bias0, weights1, bias1, weights2, bias2,
            dispatchThreadID.xy, fnb);

    float4 tapParams[4];
    float4 tapRecurrent[4];
    f16_CT2D_taps(fnb, ct2d_weights, ct2d_bias, tapParams, tapRecurrent);

    [loop]
    for (int i = 0; i < 4; ++i)
    {
        const int2 pk = localOffset[i];
        const uint2 tid = dispatchThreadID.xy * 2 + pk;

        float4 model_parameters = tapParams[i];
        float4 recurrent_f = tapRecurrent[i];

        // Recurrent Channels
        half4 recurrent = half4(recurrent_f);
        recurrent = 1.0f / (1.0f + exp(-recurrent));
        rw_recurrent_0[tid] = recurrent; // 4k tid

        float3 model_color = apply_model_filter(tid.x, tid.y, model_parameters, exposureValue); // 4k tid

#if FFX_DEBUG_VISUALIZE
        {
            float blending_factor = (1.0f / (1.0f + exp(-model_parameters.w)));
            blending_factor = 1.0f - blending_factor;
            DebugWritePredictedBlendFactor(tid, blending_factor);
        }
#endif

        // Model color (in model color space) is retained for history.
        rw_history_color[tid] = model_color; // 4k tid

        // "Final" color is converted to input color space unless RCAS is enabled
        // in which case RCAS will convert it.
        float3 final_color = model_color;

        if (shouldConvertColorspace)
            final_color.rgb = RemoveMuLaw(final_color.rgb) / exposureValue;

        final_color = clamp(final_color, 0, 64000); // Don't overflow storage (half) type.

        rw_mlsr_output_color[tid] = final_color; // 4k tid
    }
}

RWByteAddressBuffer buffer_fused_quantized_NHWC_output;
