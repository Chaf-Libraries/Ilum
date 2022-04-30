#include "../../Common.hlsli"
#include "../../Math.hlsli"
#include "../../Fetch.hlsli"

#ifndef RUNTIME
#define TASK
#define MESH
#define FRAGMENT
#endif

StructuredBuffer<Instance> instances : register(t0);
ConstantBuffer<Camera> camera : register(b2);
ConstantBuffer<CullingInfo> culling_info : register(b5);
RWStructuredBuffer<CountInfo> count_info : register(u8);

struct CSParam
{
    uint DispatchThreadID : SV_DispatchThreadID;
    uint GroupThreadID : SV_GroupThreadID;
    uint GroupID : SV_GroupID;
};

struct VertexOut
{
    float4 PositionHS : SV_Position;
};

struct PrimitiveOut
{
    uint EntityID : COLOR0;
    uint InstanceID : COLOR1;
    uint PrimitiveID : COLOR2;
    uint MeshletID : COLOR3;
};

struct Payload
{
    uint meshletIndices[32];
};

#ifdef TASK
[[vk::push_constant]]
struct
{
    uint instance_idx;
}task_push_constant;

groupshared Payload shared_payload;
[numthreads(32, 1, 1)]void ASmain(CSParam param)
{
    uint temp;
    if (param.DispatchThreadID.x == 0)
    {
        InterlockedExchange(count_info[0].meshlet_visible_count, 0, temp);
    }

    bool visible = false;

    if (param.DispatchThreadID.x < culling_info.meshlet_count)
    {
        Meshlet meshlet = meshlets[param.DispatchThreadID.x];
        Instance instance = instances[task_push_constant.instance_idx];
        
        Camera cam = camera;
        visible = meshlet.IsVisible(cam, instance.transform);
    }

    if (visible)
    {
        uint index = WavePrefixCountBits(visible);
        shared_payload.meshletIndices[index] = param.DispatchThreadID.x;
        InterlockedAdd(count_info[0].meshlet_visible_count, 1, temp);
    }

    uint visible_count = WaveActiveCountBits(visible);

    DispatchMesh(visible_count, 1, 1, shared_payload);
}
#endif

#ifdef MESH
[outputtopology("triangle")]
[numthreads(32, 1, 1)]void MSmain(CSParam param, in payload Payload pay_load, out vertices VertexOut verts[64], out indices uint3 tris[126], out primitives PrimitiveOut prims[126])
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
        
        uint vertex_index = meshlet.vertex_offset + i;
        Vertex vertex = vertices[vertex_index];

        verts[i].PositionHS = mul(camera.view_projection, mul(transform, float4(vertex.position.xyz, 1.0)));
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
        
        prims[i].EntityID = instance.entity_id;
        prims[i].InstanceID = meshlet.instance_id;
        prims[i].MeshletID = meshlet_index;
        prims[i].PrimitiveID = i;
    }
}
#endif

#ifdef FRAGMENT
uint PSmain(PrimitiveOut prims) : SV_TARGET0
{
    uint output;
    output = PackVBuffer(prims.MeshletID, prims.PrimitiveID);
    return output;
}
#endif