#include "../ShaderInterop.hpp"

ConstantBuffer<Camera> camera : register(b0);
ConstantBuffer<Instance> instances[] : register(b1);
StructuredBuffer<Meshlet> meshlets[] : register(t2);
StructuredBuffer<Vertex> vertices[] : register(t3);
StructuredBuffer<uint> meshlet_vertices[] : register(t4);
StructuredBuffer<uint> meshlet_primitives[] : register(t5);
RWStructuredBuffer<uint> debug_buffer : register(u6);

struct CSParam
{
    uint DispatchThreadID : SV_DispatchThreadID;
    uint GroupThreadID : SV_GroupThreadID;
    uint GroupID : SV_GroupID;
};

struct VertexOut
{
    float4 Position : SV_Position;
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

[[vk::push_constant]]
struct
{
    uint instance_id;
    uint meshlet_count;
} push_constants;

groupshared Payload shared_payload;
[numthreads(32, 1, 1)]
void ASmain(CSParam param)
{
    uint temp;
    if (param.DispatchThreadID.x == 0)
    {
        //InterlockedExchange(count_info[0].meshlet_visible_count, 0, temp);
    }

    bool visible = false;

    if (param.DispatchThreadID.x < push_constants.meshlet_count)
    {
        Meshlet meshlet = meshlets[push_constants.instance_id][param.DispatchThreadID.x];
        Instance instance = instances[push_constants.instance_id];
        debug_buffer[param.GroupID.x] = 1;
        //Camera cam = camera;
        //visible = meshlet.IsVisible(cam, instance.transform);
        visible = true;
    }

    if (visible)
    {
        uint index = WavePrefixCountBits(visible);
        shared_payload.meshletIndices[index] = param.DispatchThreadID.x;
        //InterlockedAdd(count_info[0].meshlet_visible_count, 1, temp);
    }

    uint visible_count = WaveActiveCountBits(visible);

    DispatchMesh(visible_count, 1, 1, shared_payload);
}

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void MSmain(CSParam param, in payload Payload pay_load, out vertices VertexOut verts[64], out indices uint3 tris[124], out primitives PrimitiveOut prims[124])
{
    uint meshlet_index = pay_load.meshletIndices[param.GroupID.x];

    if (meshlet_index >= push_constants.meshlet_count)
    {
        return;
    }

    Meshlet meshlet = meshlets[push_constants.instance_id][meshlet_index];
    float4x4 transform = instances[push_constants.instance_id].transform;

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.primitive_count);

    for (uint i = param.GroupThreadID.x; i < meshlet.vertex_count; i += 32)
    {
        uint vertex_index = meshlet_vertices[push_constants.instance_id][meshlet.vertex_offset + i];
        Vertex vertex = vertices[push_constants.instance_id][vertex_index];

        verts[i].Position = mul(camera.view_projection, mul(transform, float4(vertex.position.xyz, 1.0)));
    }

    for (i = param.GroupThreadID.x; i < meshlet.primitive_count; i += 32)
    {
        prims[i].InstanceID = push_constants.instance_id;
        prims[i].MeshletID = meshlet_index;
        prims[i].PrimitiveID = meshlet_primitives[push_constants.instance_id][i];
        
        uint v0, v1, v2;
        UnPackTriangle(meshlet_primitives[push_constants.instance_id][i+meshlet.primitive_offset], v0, v1, v2);
        
        tris[i] = uint3(v0, v1, v2);
    }
}

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

float4 PSmain(PrimitiveOut prims) : SV_TARGET0
{
    float4 output;
    uint mhash = hash(prims.MeshletID);
    float3 mcolor = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
    output = float4(mcolor, 1.0);
    return output;
}