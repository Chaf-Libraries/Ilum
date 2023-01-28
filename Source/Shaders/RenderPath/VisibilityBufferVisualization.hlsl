#include "../Common.hlsli"

Texture2D<uint> VisibilityBuffer;
Texture2D<float> DepthBuffer;
RWTexture2D<float4> InstanceID;
RWTexture2D<float4> PrimitiveID;

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
void CSmain(CSParam param)
{
    uint2 extent;
    VisibilityBuffer.GetDimensions(extent.x, extent.y);
    
    if (extent.x < param.DispatchThreadID.x || extent.y < param.DispatchThreadID.y)
    {
        return;
    }
    
    if (DepthBuffer.Load(int3(param.DispatchThreadID.xy, 0)).r > 3e38f)
    {
        InstanceID[param.DispatchThreadID.xy] = 0.f;
        PrimitiveID[param.DispatchThreadID.xy] = 0.f;
        return;
    }
    
    uint vbuffer = VisibilityBuffer.Load(int3(param.DispatchThreadID.xy, 0));
    uint instance_id = 0;
    uint primitive_id = 0;
    
    UnPackVisibilityBuffer(vbuffer, instance_id, primitive_id);
    
    uint mhash = hash(instance_id);
    InstanceID[param.DispatchThreadID.xy] = float4(float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0, 1.0);
    
    mhash = hash(primitive_id);
    PrimitiveID[param.DispatchThreadID.xy] = float4(float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0, 1.0);
}