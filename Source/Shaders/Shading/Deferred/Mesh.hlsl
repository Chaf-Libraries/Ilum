#include "../../Common.hlsli"

#ifndef RUNTIME
#define TASK
#define MESH
#define FRAGMENT
#endif

StructuredBuffer<Instance> instances : register(t0);
StructuredBuffer<Meshlet> meshlets : register(t1);
ConstantBuffer<Camera> camera : register(b2);
StructuredBuffer<Vertex> vertices : register(t3);
StructuredBuffer<uint> indices : register(t4);
ConstantBuffer<CullingInfo> culling_info : register(b5);

struct CSParam
{
    uint DispatchThreadID : SV_DispatchThreadID;
    uint GroupThreadID : SV_GroupThreadID;
    uint GroupID : SV_GroupID;
};

struct VertexOut
{
    float4 PositionHS : SV_Position;
    float3 PositionVS : POSITION0;
    float3 Normal : NORMAL0;
    uint MeshletIndex : COLOR0;
};

struct MSOutput
{
    float4 GBuffer0 : SV_Target0;
};

struct Payload
{
    uint meshletIndices[32];
};

#ifdef TASK
groupshared Payload shared_payload;
[numthreads(32, 1, 1)]
void ASmain(CSParam param)
{
    bool visible = false;
    
    if (param.DispatchThreadID.x < culling_info.meshlet_count)
    {
        visible = true;
    }
    
    if(visible)
    {
        uint index = WavePrefixCountBits(visible);
        shared_payload.meshletIndices[index] = param.DispatchThreadID.x;
    }

    uint visible_count = WaveActiveCountBits(visible);
    DispatchMesh(visible_count, 1, 1, shared_payload);
}
#endif

#ifdef MESH
[NumThreads(128, 1, 1)]
[OutputTopology("triangle")]
void MSmain(CSParam param, in payload Payload pay_load, out vertices VertexOut verts[64], out indices uint3 tris[126])
{
    uint meshlet_index = pay_load.meshletIndices[param.GroupID.x];
    
    if(meshlet_index>=culling_info.meshlet_count)
    {
        return;
    }
    
    Meshlet meshlet = meshlets[meshlet_index];
    Instance instance = instances[meshlet.instance_id];
    
    float4x4 transform = instance.transform;
    
    SetMeshOutputCounts(meshlet.vertex_count, meshlet.index_count / 3);

    if (param.GroupThreadID.x<meshlet.vertex_count)
    {
        Vertex vertex = vertices[meshlet.vertex_offset + param.GroupThreadID.x];
        
        verts[param.GroupThreadID.x].MeshletIndex = meshlet_index;
        verts[param.GroupThreadID.x].PositionHS = mul(camera.view_projection, mul(transform, float4(vertex.position.xyz, 1.0)));
        verts[param.GroupThreadID.x].PositionVS = mul(camera.view_projection, float4(vertex.position.xyz, 1.0)).xyz;
        verts[param.GroupThreadID.x].Normal = vertex.normal.xyz;
    }
    if (param.GroupThreadID.x < meshlet.index_count)
    {
        uint idx = param.GroupThreadID.x * 3;
        tris[param.GroupThreadID.x] = uint3(indices[meshlet.index_offset + idx], indices[meshlet.index_offset + idx + 1], indices[meshlet.index_offset + idx + 2]);
    }
}
#endif

#ifdef FRAGMENT
MSOutput PSmain(VertexOut input)
{
    MSOutput output;
    output.GBuffer0 = float4(input.Normal, 1.0);
    return output;
}
#endif