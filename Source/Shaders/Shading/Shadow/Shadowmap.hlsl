#include "../../Common.hlsli"
#include "../../Light.hlsli"

#ifndef RUNTIME
#define TASK
#define MESH
#define FRAGMENT
#endif

StructuredBuffer<Instance> instances : register(t0);
StructuredBuffer<Meshlet> meshlets : register(t1);
StructuredBuffer<Vertex> vertices : register(t2);
StructuredBuffer<uint> meshlet_vertices : register(t3);
StructuredBuffer<uint> meshlet_indices : register(t4);
StructuredBuffer<SpotLight> spot_lights : register(t5);
ConstantBuffer<CullingInfo> culling_info : register(b6);

[[vk::push_constant]]
struct
{
    float4x4 transform;
    uint dynamic;
    uint layer;
} push_constants;

struct CSParam
{
    uint DispatchThreadID : SV_DispatchThreadID;
    uint GroupThreadID : SV_GroupThreadID;
    uint GroupID : SV_GroupID;
};

struct VertexOut
{
    float4 Pos : SV_Position;
};

struct PrimitiveOut
{
    uint Layer : SV_RenderTargetArrayIndex;
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
    uint temp;
    
    bool visible = false;
    
    if (param.DispatchThreadID.x < culling_info.meshlet_count)
    {
        Meshlet meshlet = meshlets[param.DispatchThreadID.x];
        Instance instance = instances[meshlet.instance_id];
        
        Camera cam;
        cam.view_projection = spot_lights[push_constants.layer].view_projection;
        cam.position = spot_lights[push_constants.layer].position;
        cam.BuildFrustum();
        visible = meshlet.IsVisible(cam, instance.transform);
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
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void MSmain(CSParam param, in payload Payload pay_load, out vertices VertexOut verts[64], out indices uint3 tris[126], out primitives PrimitiveOut prims[126])
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
    
    for (uint i = param.GroupThreadID.x; i < meshlet.vertex_count; i += 32)
    {
        uint vertex_index = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + i];
        Vertex vertex = vertices[vertex_index];
        verts[i].Pos = mul(spot_lights[push_constants.layer].view_projection, mul(instance.transform, float4(vertex.position.xyz, 1.0)));
    }

    for (i = param.GroupThreadID.x; i < meshlet.index_count / 3; i += 32)
    {
        for (int j = i * 3; j < i * 3 + 3; j++)
        {
            uint a = (meshlet.meshlet_index_offset + j) / 4;
            uint b = (meshlet.meshlet_index_offset + j) % 4;
            uint idx = (meshlet_indices[a] & 0x000000ffU << (8 * b)) >> (8 * b);
            tris[i][j % 3] = idx;
            prims[i].Layer = push_constants.layer;
        }
    }
}
#endif

#ifdef FRAGMENT
void PSmain(VertexOut input)
{

}
#endif