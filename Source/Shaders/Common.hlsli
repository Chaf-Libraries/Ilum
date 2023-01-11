#ifndef COMMON_HLSLI
#define COMMON_HLSLI

#include "Math.hlsli"

#define MAX_BONE_INFLUENCE 8

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

struct RayDiff
{
    float3 dOdx;
    float3 dOdy;
    float3 dDdx;
    float3 dDdy;
    
    void Propagate(float3 direction, float t, float3 normal)
    {
        float3 dodx = dOdx + t * dDdx;
        float3 dody = dOdy + t * dDdy;

        float rcpDN = 1.0f / dot(direction, normal);
        float dtdx = -dot(dodx, normal) * rcpDN;
        float dtdy = -dot(dody, normal) * rcpDN;
        dodx += direction * dtdx;
        dody += direction * dtdy;

        dOdx = dodx;
        dOdy = dody;
    } 
};

struct View
{
    float4x4 view_matrix;
    float4x4 inv_view_matrix;
    float4x4 projection_matrix;
    float4x4 inv_projection_matrix;
    float4x4 view_projection_matrix;
    float3 position;
    uint frame_count;
    float2 viewport;
    
    void CastRay(float2 scene_uv, float2 frame_dim, out RayDesc ray, out RayDiff ray_diff)
    {
        float yaw =atan2(-view_matrix[2][2], -view_matrix[0][2]);
        float pitch = asin(-clamp(view_matrix[1][2], -1.f, 1.f));
        float3 forward = normalize(float3(cos(yaw) * cos(pitch), sin(pitch), sin(yaw) * cos(pitch)));
        float3 right = normalize(cross(forward, float3(0, 1, 0)));
        float3 up = normalize(cross(right, forward));
        float3 target = mul(inv_projection_matrix, float4(scene_uv.x, scene_uv.y, 1, 1)).xyz;

        ray.Origin = mul(inv_view_matrix, float4(0, 0, 0, 1)).xyz;
        ray.Direction = normalize(mul(inv_view_matrix, float4(target, 0)).xyz);
        ray.TMin = 0.0;
        ray.TMax = Infinity;

        float3 dir = forward + right * scene_uv.x + up * scene_uv.y;
        ray_diff.dOdx = 0;
        ray_diff.dOdy = 0;
        float dd = dot(dir, dir);
        float divd = 2.0f / (dd * sqrt(dd));
        float dr = dot(dir, right);
        float du = dot(dir, up);
        ray_diff.dDdx = ((dd * right) - (dr * dir)) * divd / frame_dim.x;
        ray_diff.dDdy = -((dd * up) - (du * dir)) * divd / frame_dim.y;
    }
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