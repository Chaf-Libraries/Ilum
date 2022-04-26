/*Texture2D BloomDownSampleIn : register(t0);
SamplerState BloomDownSampleSampler : register(s0);
RWTexture2D<float4> BloomDownSampleOut : register(u1);*/

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint GroupIndex : SV_GroupIndex;
};

Texture2D BloomDownSampleIn : register(t0);
SamplerState BloomDownSampleSampler : register(s0);
RWTexture2D<float4> BloomDownSampleOut : register(u1);

[numthreads(8, 8, 1)]
void main(CSParam param)
{
    uint parity = param.DispatchThreadID.x | param.DispatchThreadID.y;
    
    uint2 extent;
    BloomDownSampleOut.GetDimensions(extent.x, extent.y);
    
    if (param.DispatchThreadID.x >= extent.x || param.DispatchThreadID.y >= extent.y)
    {
        return;
    }
    
    float2 uv = (float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(extent);
    float3 avg_color = BloomDownSampleIn.SampleLevel(BloomDownSampleSampler, uv, 0.0).rgb;
    BloomDownSampleOut[param.DispatchThreadID.xy] = float4(avg_color, 1.0);
}