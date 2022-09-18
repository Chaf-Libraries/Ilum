#ifndef COMMON_HLSLI
#define COMMON_HLSLI

struct ViewInfo
{
    float3 position;
    float4x4 view_matrix;
    float4x4 projection_matrix;
    float4x4 view_projection_matrix;
    float2 viewport;
};

struct Vertex
{
    float3 position;
    float3 normal;
    float3 tangent;
    float2 texcoord;
};

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
};

struct MeshletBound
{
    float3 center;
    float radius;
    float3 cone_axis;
    float cone_cut_off;
};

struct InstanceData
{
    float4x4 pre_transform;

    float3 aabb_min;
    uint material_id;

    float3 aabb_max;
    uint instance_id;
};

struct Meshlet
{
    MeshletBound bound;
    uint indices_offset;
    uint indices_count;
    uint vertices_offset;
    uint vertices_count;
    uint meshlet_vertices_offset;
    uint meshlet_indices_offset;
};

#endif