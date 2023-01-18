#ifndef MATERIAL_RESOURCE_HLSLI
#define MATERIAL_RESOURCE_HLSLI

#include "Interaction.hlsli"

Texture2D<float4> Textures[] : register(s996);
SamplerState Samplers[] : register(s997);
StructuredBuffer<uint> MaterialOffsets : register(t998);
ByteAddressBuffer MaterialBuffer : register(t999);

float4 SampleTexture2D(uint texture_id, uint sampler_id, float2 uv, float2 duvdx, float2 duvdy)
{
    if (texture_id == ~0U)
    {
        return 0.f;
    }
    
#ifdef RASTERIZATION_PIPELINE
    return Textures[texture_id].Sample(Samplers[sampler_id], uv);
#else
    return Textures[texture_id].SampleGrad(Samplers[sampler_id], uv, duvdx, duvdy);
#endif
}

template<typename T>
T GetMaterialData(uint material_id)
{
    if (material_id == ~0U)
    {
        T data;
        return data;
    }
    return MaterialBuffer.Load<T>(MaterialOffsets[material_id]);
}

void SetNormalMap(inout SurfaceInteraction interaction, float3 normal_vector)
{
    Frame frame;
    frame.FromZ(interaction.isect.n);
    float3x3 TBN = float3x3(frame.x, frame.y, frame.z);
    normal_vector = normalize(normal_vector * 2.0 - 1.0);
    float3 normal = normalize(mul(normal_vector, TBN));
    interaction.isect.n = dot(interaction.isect.n, normal) <= 0.0 ? normal : -normal;
}

#endif