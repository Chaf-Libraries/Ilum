#include "../ShaderInterop.hpp"

ConstantBuffer<Camera> camera : register(b0);
ConstantBuffer<Instance> instances[] : register(b1);
StructuredBuffer<Meshlet> meshlets[] : register(t2);
StructuredBuffer<Vertex> vertices[] : register(t3);
StructuredBuffer<uint> meshlet_vertices[] : register(t4);
StructuredBuffer<uint> meshlet_primitives[] : register(t5);
#ifdef ALPHA_TEST
ConstantBuffer<Material> materials[] : register(b6);
Texture2D<float4> texture_array[] : register(t7);
SamplerState texture_sampler : register(s8);
#endif

struct CSParam
{
    uint DispatchThreadID : SV_DispatchThreadID;
    uint GroupThreadID : SV_GroupThreadID;
    uint GroupID : SV_GroupID;
};

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
        Meshlet meshlet = meshlets[instances[push_constants.instance_id].mesh][param.DispatchThreadID.x];
        //debug_buffer[param.GroupID.x] = param.GroupID.x;
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

    float4x4 transform = instances[push_constants.instance_id].transform;
    
    Meshlet meshlet = meshlets[instances[push_constants.instance_id].mesh][meshlet_index];

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.primitive_count);

    for (uint i = param.GroupThreadID.x; i < meshlet.vertex_count; i += 32)
    {
        uint vertex_index = meshlet_vertices[instances[push_constants.instance_id].mesh][meshlet.vertex_offset + i];
        Vertex vertex = vertices[instances[push_constants.instance_id].mesh][vertex_index];

        verts[i].Position = mul(camera.view_projection, mul(transform, float4(vertex.position.xyz, 1.0)));
        verts[i].Texcoord = vertex.texcoord.xy;
    }

    for (i = param.GroupThreadID.x; i < meshlet.primitive_count; i += 32)
    {
        prims[i].InstanceID = instances[push_constants.instance_id].mesh;
        prims[i].MeshletID = meshlet_index;
        prims[i].PrimitiveID = meshlet_primitives[instances[push_constants.instance_id].mesh][i];
        
        uint v0, v1, v2;
        UnPackTriangle(meshlet_primitives[instances[push_constants.instance_id].mesh][i + meshlet.primitive_offset], v0, v1, v2);
        
        tris[i] = uint3(v0, v1, v2);
    }
}

uint PSmain(VertexOut verts, PrimitiveOut prims) : SV_TARGET0
{
#ifdef ALPHA_TEST
    float alpha = materials[instances[push_constants.instance_id].material].albedo_factor.a;
    uint albedo_tex_id = materials[instances[push_constants.instance_id].material].albedo_texture;
    if (albedo_tex_id < 1024)
    {
        float4 albedo = texture_array[albedo_tex_id].Sample(texture_sampler, verts.Texcoord);
       alpha *= albedo.a;
    }
    if (alpha < materials[instances[push_constants.instance_id].material].alpha_cut_off)
    {
        discard;
    }
#endif    
    uint vbuffer = PackVBuffer(prims.InstanceID, prims.MeshletID, prims.PrimitiveID);
    return vbuffer;
}