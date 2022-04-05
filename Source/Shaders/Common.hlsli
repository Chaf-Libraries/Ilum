#ifndef __COMMON_HLSL__
#define __COMMON_HLSL__

// Camera Data
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

// Per Instance Data
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

// Per Meshlet Data
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

// Info for Culling
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

// Indirect Draw Command
struct DrawIndexedIndirectCommand
{
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

// Infod for Count
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

// DIrectional Light Source
struct DirectionalLight
{
    float4 split_depth;
    float4x4 view_projection[4];
    float3 color;
    float intensity;
    float3 direction;

    int shadow_mode; // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
    float filter_scale;
    int filter_sample;
    int sample_method; // 0 - Uniform, 1 - Poisson Disk
    float light_size;

    float3 position;
};

// Point Light Source
struct PointLight
{
    float3 color;
    float intensity;
    float3 position;
    float constant;
    float linear_;
    float quadratic;

    int shadow_mode; // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
    float filter_scale;
    int filter_sample;
    int sample_method; // 0 - Uniform, 1 - Poisson Disk
    float light_size;
};

// Spot Light Source
struct SpotLight
{
    float4x4 view_projection;
    float3 color;
    float intensity;
    float3 position;
    float cut_off;
    float3 direction;
    float outer_cut_off;

    int shadow_mode; // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
    float filter_scale;
    int filter_sample;
    int sample_method; // 0 - Uniform, 1 - Poisson Disk
    float light_size;
};

#endif