#include "ShaderInterop.hpp"

Texture2D<uint> VBuffer : register(t0);
RWTexture2D<float4> InstanceID : register(u1);
RWTexture2D<float4> PrimitiveID : register(u2);
RWTexture2D<float4> MeshletID : register(u3);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

uint hash(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    uint2 extent;
    VBuffer.GetDimensions(extent.x, extent.y);
    
    if (extent.x < param.DispatchThreadID.x || extent.y < param.DispatchThreadID.y)
    {
        return;
    }
    
    uint vbuffer = VBuffer.Load(int3(param.DispatchThreadID.xy, 0));
    uint instance_id = 0;
    uint primitive_id = 0;
    uint meshlet_id = 0;
    UnPackVBuffer(vbuffer, instance_id, meshlet_id, primitive_id);
    uint mhash = hash(instance_id);
    InstanceID[param.DispatchThreadID.xy] = float4(float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0, 1.0);
    mhash = hash(primitive_id);
    PrimitiveID[param.DispatchThreadID.xy] = float4(float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0, 1.0);
    mhash = hash(meshlet_id);
    MeshletID[param.DispatchThreadID.xy] = float4(float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0, 1.0);
}