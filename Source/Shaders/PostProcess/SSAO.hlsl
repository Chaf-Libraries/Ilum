#include "../Common.hlsli"
#include "../Random.hlsli"

#define KERNAL_SIZE 64
#define RADIUS 0.5
#define NOISE_DIM 4

struct Sample
{
    float4 samples[KERNAL_SIZE];
};

Texture2D<float4> Normal;
Texture2D<float4> PositionDepth;
RWTexture2D<float> SSAOMap;
SamplerState TexSampler;
ConstantBuffer<View> ViewBuffer;
Texture2D<float4> NoiseTexture;
ConstantBuffer<Sample> SampleBuffer;

[numthreads(8, 8, 1)]
void SSAO(CSParam param)
{
    uint2 extent;
    SSAOMap.GetDimensions(extent.x, extent.y);
    
    if (param.DispatchThreadID.x >= extent.x ||
        param.DispatchThreadID.y >= extent.y)
    {
        return;
    }
    
    float2 uv = (float2(param.DispatchThreadID.xy) + 0.5f) / float2(extent);
    
    float3 position = PositionDepth[param.DispatchThreadID.xy].rgb;
    
    if (IsBlack(position))
    {
        SSAOMap[param.DispatchThreadID.xy] = 1.f;
        return;
    }
    
    position = mul(ViewBuffer.view_matrix, float4(position, 1.f)).xyz;
    float3 normal = Normal[param.DispatchThreadID.xy].rgb * 2.f - 1.f;
    normal = mul((float3x3) ViewBuffer.view_matrix, normal).xyz;
    
    PCGSampler rng;
    rng.Init(extent, param.DispatchThreadID.xy, 0);
    
    uint2 noise_extent;
    NoiseTexture.GetDimensions(noise_extent.x, noise_extent.y);
    
    const float2 noise_uv = float2(float(extent.x) / float(noise_extent.x), float(extent.y) / (noise_extent.y)) * uv;
    float3 random_vec = NoiseTexture.SampleLevel(TexSampler, noise_uv, 0).xyz * 2.0 - 1.0;
    
    float3 tangent = normalize(random_vec - normal * dot(random_vec, normal));
    float3 bitangent = cross(tangent, normal);
    float3x3 TBN = transpose(float3x3(tangent, bitangent, normal));
    
    const float bias = 0.025f;
    float occlusion = 0.f;
    float3 color = 0.f;
    
    for (uint i = 0; i < KERNAL_SIZE; i++)
    {
        float3 sample_pos = mul(TBN, SampleBuffer.samples[i].xyz);
        sample_pos = position + sample_pos * RADIUS;
        
        float4 offset = float4(sample_pos, 1.f);
        offset = mul(ViewBuffer.projection_matrix, offset);
        offset.y = -offset.y;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5f + 0.5f;

        float sample_depth = PositionDepth.SampleLevel(TexSampler, offset.xy, 0).w;
        float range = smoothstep(0.f, 1.f, RADIUS / abs(position.z - sample_depth));
        
        occlusion += (sample_depth >= sample_pos.z ? 1.0f : 0.0f) * range;
    }
    occlusion = 1.f - (occlusion / float(KERNAL_SIZE));
    SSAOMap[param.DispatchThreadID.xy] = occlusion;
}

groupshared float cache[8][8][9];

[numthreads(8, 8, 1)]
void SSAOBlur(CSParam param)
{
    uint2 extent;
    SSAOMap.GetDimensions(extent.x, extent.y);
    
    if (param.DispatchThreadID.x >= extent.x ||
        param.DispatchThreadID.y >= extent.y)
    {
        return;
    }
    
    uint idx = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            cache[param.GroupThreadID.x][param.GroupThreadID.y][idx] = SSAOMap[int2(clamp(param.DispatchThreadID.x + i, 0, extent.x), clamp(param.DispatchThreadID.y + j, 0, extent.y))];
        }
    }
    
    GroupMemoryBarrierWithGroupSync();

    // Mean filter
    float ssao = 0.f;
    for (int i = 0; i < 9; i++)
    {
        ssao += cache[param.GroupThreadID.x][param.GroupThreadID.y][i];
    }
    
    SSAOMap[param.GroupThreadID.xy] = ssao / 9.f;
}