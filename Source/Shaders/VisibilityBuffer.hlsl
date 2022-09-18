#include "Common.hlsli"

//ConstantBuffer<ViewInfo> View : register(b0);
//StructuredBuffer<Vertex> VertexBuffer[] : register(t1);
//StructuredBuffer<uint> IndexBuffer[] : register(t2);
//StructuredBuffer<uint> MeshletVertexBuffer[] : register(t3);
//StructuredBuffer<uint> MeshletIndexBuffer[] : register(t4);
StructuredBuffer<InstanceData> InstanceBuffer : register(t5);

struct VertexOut
{
    float4 Position : SV_Position;
    float2 Texcoord : TEXCOORD0;
};

struct PrimitiveOut
{
    uint InstanceID : COLOR1;
    uint PrimitiveID : COLOR2;
    uint MeshletID : COLOR3;
};

struct PayLoad
{
    uint meshletIndices[32][32];
};

groupshared PayLoad s_payload;

[numthreads(32, 32, 1)]
void ASmain(CSParam param)
{
    bool visible = true;
    
    uint meshlet_idx = param.DispatchThreadID.x;
    uint instance_idx = param.DispatchThreadID.y;
    
    InstanceData instance = InstanceBuffer[instance_idx];
    
    
    uint index = WavePrefixCountBits(visible);
    s_payload.meshletIndices[param.DispatchThreadID.y][index] = param.DispatchThreadID.x;
    
    uint visible_count = WaveActiveCountBits(visible);
    
    DispatchMesh(visible_count, instance_idx, 1, s_payload);
}

[outputtopology("triangle")]
[numthreads(32, 32, 1)]
void MSmain(CSParam param, in payload PayLoad pay_load, out vertices VertexOut verts[64], out indices uint3 tris[124], out primitives PrimitiveOut prims[124])
{
    uint meshlet_index = pay_load.meshletIndices[param.GroupID.y][param.GroupID.x];
    
    
}

uint PSmain(VertexOut verts, PrimitiveOut prims) : SV_TARGET0
{
    return 0;
}