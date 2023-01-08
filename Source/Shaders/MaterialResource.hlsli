#ifndef MATERIAL_RESOURCE_HLSLI
#define MATERIAL_RESOURCE_HLSLI

Texture2D<float4> Textures;
SamplerState Samplers;
StructuredBuffer<uint> MaterialOffsets;
ByteAddressBuffer MaterialBuffer;

template<typename T>
T GetMaterialBuffer(uint material_id)
{
    MaterialBuffer.Load<T>(MaterialOffsets[material_id]);
}

#endif