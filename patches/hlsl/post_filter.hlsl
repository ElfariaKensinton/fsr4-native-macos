// Native INT8 model parameters are already in scratch. One thread filters
// one output pixel; a 16x16 group still matches the host's dispatch geometry.
// Use the compiler's shadow include tree, including its loop policy.
#include <post_common.hlsli>

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          uint3 local : SV_GroupThreadID,
          uint3 group : SV_GroupID)
{
    const float exposureValue = LoadExposureValue();
    cache_filter_color(local.y * 16 + local.x, group.xy * 16, 256, exposureValue);
    if (tid.x >= width || tid.y >= height)
        return;

    const uint2 pd = ScratchBuffer.Load2(FSR4_POST_EXT_CT2D_OFF + (tid.y * width + tid.x) * 8);
    const float4 model_parameters = Unpack4h(pd);
    const float3 model_color = apply_model_filter(tid.x, tid.y, model_parameters, exposureValue);
#if FFX_DEBUG_VISUALIZE
    DebugWritePredictedBlendFactor(tid.xy, 1.0f - 1.0f / (1.0f + exp(-model_parameters.w)));
#endif
    rw_history_color[tid.xy] = model_color;
    float3 final_color = model_color;
    if (!rcas_enabled)
        final_color = RemoveMuLaw(final_color) / exposureValue;
    rw_mlsr_output_color[tid.xy] = clamp(final_color, 0, 64000);
}
