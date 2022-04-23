/*#include "../../Common.hlsli"
#include "../../Light.hlsli"

StructuredBuffer<Instance> instances : register(t0);
StructuredBuffer<PointLight> point_lights : register(t1);

[[vk::push_constant]]
struct
{
    float4x4 transform;
    float4x4 view_projection;
    float3 light_pos;
    uint dynamic;
    uint light_id;
    uint face_id;
    float depth_bias;
} push_constants;

struct VSInput
{
    uint InstanceID : SV_InstanceID;
    [[vk::location(0)]] float3 Pos : POSITION0;
    [[vk::location(1)]] float2 UV : TEXCOORD0;
    [[vk::location(2)]] float3 Normal : COLOR0;
};

struct VSOutput
{
    float4 Pos_ : SV_Position;
    uint Layer : SV_RenderTargetArrayIndex;
    float4 Pos : POSITION0;
    float3 LightPos : POSITION1;
    float DepthBias : COLOR0;
};

struct PSInput
{
    float4 Pos : POSITION0;
    float3 LightPos : POSITION1;
    float DepthBias : COLOR0;
};

struct PSOutput
{
    float depth : SV_Depth;
};

VSOutput VSmain(VSInput input)
{
    VSOutput output;
    float4x4 trans = push_constants.dynamic == 1 ? push_constants.transform : instances[input.InstanceID].transform;
    output.Layer = push_constants.light_id * 6 + push_constants.face_id;
    output.Pos_ = mul(push_constants.view_projection, mul(trans, float4(input.Pos, 1.0)));
    
    output.Pos = mul(trans, float4(input.Pos, 1.0));
    output.LightPos = push_constants.light_pos;
    output.DepthBias = push_constants.depth_bias;
    
    return output;
}

PSOutput PSmain(PSInput input)
{
    PSOutput output;
    output.depth = (length(input.Pos.xyz - input.LightPos.xyz) + input.DepthBias);
    return output;
}*/

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
StructuredBuffer<PointLight> point_lights : register(t5);
ConstantBuffer<CullingInfo> culling_info : register(b6);

[[vk::push_constant]]
struct
{
    float4x4 view_projection;
    uint light_id;
    uint face_id;
    float depth_bias;
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
    float3 PosVS : POSITIONT0;
    float3 LightPos : POSITIONT1;
    float DepthBias : NORMAL0;
};

struct PSOutput
{
    float depth : SV_Depth;
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
        cam.view_projection = push_constants.view_projection;
        cam.position = point_lights[push_constants.light_id].position;
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
        verts[i].Pos = mul(push_constants.view_projection, mul(instance.transform, float4(vertex.position.xyz, 1.0)));
        verts[i].PosVS = mul(instance.transform, float4(vertex.position.xyz, 1.0)).xyz;
        verts[i].DepthBias = push_constants.depth_bias;
        verts[i].LightPos = point_lights[push_constants.light_id].position;
    }

    for (i = param.GroupThreadID.x; i < meshlet.index_count / 3; i += 32)
    {
        for (int j = i * 3; j < i * 3 + 3; j++)
        {
            uint a = (meshlet.meshlet_index_offset + j) / 4;
            uint b = (meshlet.meshlet_index_offset + j) % 4;
            uint idx = (meshlet_indices[a] & 0x000000ffU << (8 * b)) >> (8 * b);
            tris[i][j % 3] = idx;
            prims[i].Layer = 6 * push_constants.light_id + push_constants.face_id;
        }
    }
}
#endif

#ifdef FRAGMENT
PSOutput PSmain(VertexOut input)
{
    PSOutput output;
    output.depth = (length(input.PosVS.xyz - input.LightPos.xyz) + input.DepthBias) / 100.0;
    return output;
}
#endif