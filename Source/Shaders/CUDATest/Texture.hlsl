RWTexture2D<float4> Result;
Texture2D Texture;
SamplerState Sampler;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
};

[numthreads(8, 8, 1)]
void MainCS(CSParam param)
{
    if (param.DispatchThreadID.x > 100 || param.DispatchThreadID.y > 100)
    {
        return;
    }
    
    float4 x00 = Texture.SampleLevel(Sampler, float2(param.DispatchThreadID.xy - uint2(-1, -1)) / 100.f, 0.f);
    float4 x01 = Texture.SampleLevel(Sampler, float2(param.DispatchThreadID.xy - uint2(-1, 0)) / 100.f, 0.f);
    float4 x02 = Texture.SampleLevel(Sampler, float2(param.DispatchThreadID.xy - uint2(-1, 1)) / 100.f, 0.f);
    float4 x10 = Texture.SampleLevel(Sampler, float2(param.DispatchThreadID.xy - uint2(0, -1)) / 100.f, 0.f);
    float4 x11 = Texture.SampleLevel(Sampler, float2(param.DispatchThreadID.xy - uint2(0, 0)) / 100.f, 0.f);
    float4 x12 = Texture.SampleLevel(Sampler, float2(param.DispatchThreadID.xy - uint2(0, 1)) / 100.f, 0.f);
    float4 x20 = Texture.SampleLevel(Sampler, float2(param.DispatchThreadID.xy - uint2(1, -1)) / 100.f, 0.f);
    float4 x21 = Texture.SampleLevel(Sampler, float2(param.DispatchThreadID.xy - uint2(1, 0)) / 100.f, 0.f);
    float4 x22 = Texture.SampleLevel(Sampler, float2(param.DispatchThreadID.xy - uint2(1, 1)) / 100.f, 0.f);
    
    float4 result = 1.f / 12.f * (x00 + 2 * x01 + x02 + 2 * x10 - 12 * x11 + 2 * x12 + x20 + 2 * x21 + x22);
    Result[param.DispatchThreadID.xy] = float4(result.xyz, 1.f);
}