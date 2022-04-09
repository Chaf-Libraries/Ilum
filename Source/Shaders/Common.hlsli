#ifndef __COMMON_HLSL__
#define __COMMON_HLSL__

#include "Constants.hlsli"

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
    
    RayDesc CastRay(float2 screen_coords)
    {
        RayDesc ray;

        float4 target = mul(inverse_projection, float4(screen_coords.x, screen_coords.y, 1, 1));

        ray.Origin = mul(inverse_view, float4(0, 0, 0, 1)).xyz;
        ray.Direction = mul(inverse_view, float4(normalize(target.xyz), 0)).xyz;
        ray.TMin = 0.0;
        ray.TMax = Infinity;

        return ray;
    }
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

#define TEXTURE_BASE_COLOR 0
#define TEXTURE_NORMAL 1
#define TEXTURE_METALLIC 2
#define TEXTURE_ROUGHNESS 3
#define TEXTURE_EMISSIVE 4
#define TEXTURE_AO 5
#define TEXTURE_DISPLACEMENT 6
#define TEXTURE_MAX_NUM 7

#define MAX_TEXTURE_ARRAY_SIZE 1024

#define Material_CookTorrance 0
#define Material_Disney 1
#define Material_Matte 2
#define Material_Plastic 3
#define Material_Metal 4
#define Material_Mirror 5
#define Material_Substrate 6
#define Material_Glass 7

struct MaterialData
{
    float4 base_color;

    float3 emissive_color;
    float emissive_intensity;

    float displacement;
    float subsurface;
    float metallic;
    float specular;

    float specular_tint;
    float roughness;
    float anisotropic;
    float sheen;

    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;
    float specular_transmission;

    float diffuse_transmission;
    uint textures[TEXTURE_MAX_NUM];

    float3 data;
    uint material_type;
    
    float refraction;
    float flatness;
    float thin;
};

struct Material
{
    float4 base_color;
    float3 emissive;
    float subsurface;
    float metallic;
    float specular;
    float specular_tint;
    float roughness;
    float anisotropic;
    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;
    float specular_transmission;
    float diffuse_transmission;
    float refraction;
    float flatness;
    float thin;
    float3 data;
    uint material_type;
};

struct Vertex
{
    float4 position;
    float4 uv;
    float4 normal;
};

struct RayPayload
{
    uint seed;
    float hitT;
    int primitiveID;
    int instanceID;
    float2 baryCoord;
    float4x3 objectToWorld;
    float4x3 worldToObject;
};

struct ShadowPayload
{
    bool visibility;
};

struct ShadeState
{
    float3 normal;
    float3 geom_normal;
    float3 position;
    float2 tex_coord;
    float3 tangent_u;
    float3 tangent_v;
    uint matIndex;
};

struct Interaction
{
    int depth;
    float eta;

    float3 position;
    float3 normal;
    float3 ffnormal;
    float3 tangent;
    float3 bitangent;
    float2 texCoord;
    float3 wo;

    Material material;
    
    void CreateCoordinateSystem()
    {
        tangent = normalize(((abs(ffnormal.z) > 0.99999f) ? float3(-ffnormal.x * ffnormal.y, 1.0f - ffnormal.y * ffnormal.y, -ffnormal.y * ffnormal.z) :
                                            float3(-ffnormal.x * ffnormal.z, -ffnormal.y * ffnormal.z, 1.0f - ffnormal.z * ffnormal.z)));
        bitangent = cross(tangent, ffnormal);
    }
    
    float3 WorldToLocal(float3 w)
    {
        return float3(dot(w, tangent), dot(w, bitangent), dot(w, normal));
    }
    
    float3 LocalToWorld(float3 w)
    {
        return tangent * w.x + bitangent * w.y + normal * w.z;
    }
    
    float3 WorldToLocalDir(float3 w)
    {
        return float3(dot(w, tangent), dot(w, bitangent), dot(w, ffnormal));
    }
    
    float3 LocalToWorldDir(float3 w)
    {
        return tangent * w.x + bitangent * w.y + ffnormal * w.z;
    }
};
#endif