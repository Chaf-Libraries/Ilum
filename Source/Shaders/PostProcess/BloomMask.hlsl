#include "../Common.hlsli"

RWTexture2D<float4> BloomMaskIn : register(u0);
RWTexture2D<float4> BloomMaskOut : register(u1);

[[vk::push_constant]]
struct
{
    float threshold;
} mask_push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    uint2 extent;
    BloomMaskIn.GetDimensions(extent.x, extent.y);
    
    if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
    {
        return;
    }
    
    float3 color = BloomMaskIn.Load(uint3(param.DispatchThreadID.xy, 0)).rgb;
    float lum = Luminance(color);

    BloomMaskOut[param.DispatchThreadID.xy] = float4(clamp(lum - mask_push_constants.threshold, 0.0, 1.0) * color, 1.0);
}