#include "../../Common.hlsli"
#include "../../Math.hlsli"

#ifndef RUNTIME
#define TASK
#define MESH
#define FRAGMENT
#endif

StructuredBuffer<Instance> instances : register(t0);
StructuredBuffer<Meshlet> meshlets : register(t1);
ConstantBuffer<Camera> camera : register(b2);
StructuredBuffer<Vertex> vertices : register(t3);
//StructuredBuffer<uint> indices : register(t4);
ConstantBuffer<CullingInfo> culling_info : register(b5);
StructuredBuffer<uint> meshlet_vertices : register(t6);
StructuredBuffer<uint> meshlet_indices : register(t7);
RWStructuredBuffer<uint> debug : register(u8);

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
    float3 Color : COLOR0;
    uint MeshletIndex : COLOR1;
};

struct MSOutput
{
    float4 GBuffer0 : SV_Target0;
};

struct Payload
{
    uint meshletIndices[32];
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

#ifdef TASK
groupshared Payload shared_payload;
[numthreads(32, 1, 1)]
void ASmain(CSParam param)
{
    bool visible = false;
    
    if (param.DispatchThreadID.x < culling_info.meshlet_count)
    {
        Meshlet meshlet = meshlets[param.DispatchThreadID.x];
        Instance instance = instances[meshlet.instance_id];
        
        float4x4 transform = instance.transform;
        
        BoundingSphere bound = meshlet.bound;
        bound.Transform(transform);
        
        Camera cam = camera;
        visible = bound.IsVisible(cam);
        //visible = true;
    }
    
    if (visible)
    {
        uint index = WavePrefixCountBits(visible);
        shared_payload.meshletIndices[index] = param.DispatchThreadID.x;
    }

    uint visible_count = WaveActiveCountBits(visible);
    DispatchMesh(visible_count, 1, 1, shared_payload);
}
#endif

#ifdef MESH
/*[[vk::push_constant]]
struct
{
    int primitive_count;
} push_constants;*/

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void MSmain(CSParam param, in payload Payload pay_load, out vertices VertexOut verts[64], out indices uint3 tris[126])
{
    
    uint meshlet_index = pay_load.meshletIndices[param.GroupID.x];
    
    if (meshlet_index >= culling_info.meshlet_count)
    {
        return;
    }
    
    Meshlet meshlet = meshlets[meshlet_index];
    Instance instance = instances[meshlet.instance_id];
    
    float4x4 transform = instance.transform;
    
    SetMeshOutputCounts(meshlet.vertex_count, meshlet.index_count / 3);
    //SetMeshOutputCounts(meshlet.vertex_count, push_constants.primitive_count);
    
    for (uint i = param.GroupThreadID.x; i < meshlet.vertex_count; i += 32)
    {
        Vertex vertex = vertices[meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + i]];
        verts[i].PositionHS = mul(camera.view_projection, mul(transform, float4(vertex.position.xyz, 1.0)));
        uint mhash = hash(meshlet_index);
        verts[i].Color = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
        
        if (meshlet_index==1)
        {
            debug[i] = meshlet.vertex_offset;
            //+meshlet_vertices[meshlet.meshlet_vertex_offset + i];
        }
        
        
    }

    for (i = param.GroupThreadID.x; i < meshlet.index_count / 3; i += 32)
    {
        for (int j = i * 3; j < i * 3 + 3; j++)
        {
            uint a = (meshlet.meshlet_index_offset + j) / 4;
            uint b = (meshlet.meshlet_index_offset + j) % 4;
            uint idx = (meshlet_indices[a] & 0x000000ffU << (8 * b)) >> (8 * b);
            tris[i][j % 3] = idx;
            
        }
        
    }
}
#endif

#ifdef FRAGMENT
MSOutput PSmain(VertexOut input)
{
    MSOutput output;
    output.GBuffer0 = float4(input.Color, 1.0);
    return output;
}
#endif