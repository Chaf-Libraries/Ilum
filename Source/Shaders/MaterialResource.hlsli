#ifndef MATERIAL_RESOURCE_HLSLI
#define MATERIAL_RESOURCE_HLSLI

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

#endif