#ifndef __COMMON_HLSL__
#define __COMMON_HLSL__

struct Camera
{
    float4x4 view_projection;
    float4x4 last_view_projection;
    float4x4 inverse_view;
    float4x4 inverse_projection;
    float4 frustum[6];
    float3 position;
    uint frame_count;
};

struct Instance
{
    float4x4 transform;
    float4x4 last_transform;

    float3 bbox_min;
    uint entity_id;

    float3 bbox_max;
    uint material_id;

    uint vertex_offset;
    uint index_offset;
    uint index_count;
};

struct Meshlet
{
    uint instance_id;
    uint vertex_offset;
    uint index_offset;
    uint index_count;

    float3 center;
    float radius;

    float3 cone_apex;
    float cone_cutoff;

    float3 cone_axis;
};

struct CullingInfo
{
    float4x4 view;

    float4x4 last_view;

    float P00;
    float P11;
    float znear;
    float zfar;

    float zbuffer_width;
    float zbuffer_height;
    uint meshlet_count;
    uint instance_count;
};

struct DrawIndexedIndirectCommand
{
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

struct CountInfo
{
    uint actual_draw;
    uint total_draw;
    uint meshlet_visible_count;
    uint instance_visible_count;
    uint meshlet_invisible_count;
    uint instance_invisible_count;
    uint meshlet_total_count;
    uint instance_total_count;
};

#endif