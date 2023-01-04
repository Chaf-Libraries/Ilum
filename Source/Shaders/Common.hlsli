#ifndef COMMON_HLSLI
#define COMMON_HLSLI

#include "Math.hlsli"

#define MAX_BONE_INFLUENCE 8

struct View
{
    float4x4 view_matrix;
    float4x4 inv_view_matrix;
    float4x4 projection_matrix;
    float4x4 inv_projection_matrix;
    float4x4 view_projection_matrix;
    float3 position;
    uint frame_count;
    
    RayDesc CastRay(float2 sceneUV)
    {
        RayDesc ray;

        float4 target = mul(inv_projection_matrix, float4(sceneUV.x, sceneUV.y, 1, 1));

        ray.Origin = mul(inv_view_matrix, float4(0, 0, 0, 1)).xyz;
        ray.Direction = mul(inv_view_matrix, float4(normalize(target.xyz), 0)).xyz;
        ray.TMin = 0.0;
        ray.TMax = Infinity;

        return ray;
    }
};

struct Vertex
{
    float3 position;
    float3 normal;
    float3 tangent;
    float2 texcoord0;
    float2 texcoord1;
};

struct SkinnedVertex
{
    float3 position;
    float3 normal;
    float3 tangent;

    float2 texcoord0;
    float2 texcoord1;

    int bones[8];
    float weights[8];
};

struct Meshlet
{
    float3 center;
    float radius;
    
    float3 cone_axis;
    float cone_cutoff;

    uint data_offset;
    uint vertex_offset;
    uint vertex_count;
    uint triangle_count;
};

struct Instance
{
    float4x4 transform;

    uint mesh_id;
    uint material_id;
    uint animation_id;
};

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
    uint GroupIndex : SV_GroupIndex;
};

void UnPackTriangle(uint encode, out uint v0, out uint v1, out uint v2)
{
    v0 = encode & 0xff;
    v1 = (encode >> 8) & 0xff;
    v2 = (encode >> 16) & 0xff;
}

uint PackVisibilityBuffer(uint instance_id, uint primitive_id)
{
    // Instance ID 8
    // Primitive ID 24
    uint vbuffer = 0;
    vbuffer += instance_id & 0xff;
    vbuffer += (primitive_id & 0xffffff) << 8;
    return vbuffer;
}

void UnPackVisibilityBuffer(uint visibility_buffer, out uint instance_id, out uint primitive_id)
{
    // Instance ID 8
    // Primitive ID 24
    instance_id = visibility_buffer & 0xff;
    primitive_id = (visibility_buffer >> 8) & 0xffffff;
}

#endif