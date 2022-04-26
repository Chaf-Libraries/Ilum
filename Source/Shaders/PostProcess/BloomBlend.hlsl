Texture2D BloomBlendBloom : register(t0);
SamplerState BloomBlendSampler : register(s0);
Texture2D BloomBlendIn : register(t1);
RWTexture2D<float4> BloomBlendOut : register(u2);

[[vk::push_constant]]
struct
{
    float intensity;
    uint enable;
} blend_push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

[numthreads(8, 8, 1)]
void main(CSParam param)
{
    uint2 extent;
    BloomBlendOut.GetDimensions(extent.x, extent.y);
    
    if (param.DispatchThreadID.x >= extent.x || param.DispatchThreadID.y >= extent.y)
    {
        return;
    }
    
    float2 uv = (float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(extent);
    
    if (blend_push_constants.enable == 1)
    {
        BloomBlendOut[param.DispatchThreadID.xy] = float4(BloomBlendIn.SampleLevel(BloomBlendSampler, uv, 0.0).rgb + blend_push_constants.intensity * BloomBlendBloom.SampleLevel(BloomBlendSampler, uv, 0.0).rgb, 1.0);
    }
    else
    {
        BloomBlendOut[param.DispatchThreadID.xy] = float4(BloomBlendIn.SampleLevel(BloomBlendSampler, uv, 0.0).rgb, 1.0);
    }
}