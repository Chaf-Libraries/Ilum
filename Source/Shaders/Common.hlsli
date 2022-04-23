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
    
    void BuildFrustum()
    {
        // Left
        frustum[0].x = view_projection[0].w + view_projection[0].x;
        frustum[0].y = view_projection[1].w + view_projection[1].x;
        frustum[0].z = view_projection[2].w + view_projection[2].x;
        frustum[0].w = view_projection[3].w + view_projection[3].x;

	    // Right
        frustum[1].x = view_projection[0].w - view_projection[0].x;
        frustum[1].y = view_projection[1].w - view_projection[1].x;
        frustum[1].z = view_projection[2].w - view_projection[2].x;
        frustum[1].w = view_projection[3].w - view_projection[3].x;

	    // Top
        frustum[2].x = view_projection[0].w - view_projection[0].y;
        frustum[2].y = view_projection[1].w - view_projection[1].y;
        frustum[2].z = view_projection[2].w - view_projection[2].y;
        frustum[2].w = view_projection[3].w - view_projection[3].y;

	    // Bottom
        frustum[3].x = view_projection[0].w + view_projection[0].y;
        frustum[3].y = view_projection[1].w + view_projection[1].y;
        frustum[3].z = view_projection[2].w + view_projection[2].y;
        frustum[3].w = view_projection[3].w + view_projection[3].y;

	    // Near
        frustum[4].x = view_projection[0].w + view_projection[0].z;
        frustum[4].y = view_projection[1].w + view_projection[1].z;
        frustum[4].z = view_projection[2].w + view_projection[2].z;
        frustum[4].w = view_projection[3].w + view_projection[3].z;

	    // Far
        frustum[5].x = view_projection[0].w - view_projection[0].z;
        frustum[5].y = view_projection[1].w - view_projection[1].z;
        frustum[5].z = view_projection[2].w - view_projection[2].z;
        frustum[5].w = view_projection[3].w - view_projection[3].z;

        for (uint i = 0; i < 6; i++)
        {
            float len = length(frustum[i]);
            frustum[i] /= len;
            frustum[i].w /= len;
        }
    }
};

struct BoundingSphere
{
    float3 center;
    float radius;
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
    BoundingSphere bound;

    float3 cone_apex;
    float cone_cutoff;

    float3 cone_axis;
    uint instance_id;
    
    uint vertex_count;
    uint vertex_offset;
    uint index_offset;
    uint index_count;

    uint meshlet_vertex_offset;
    uint meshlet_index_offset;
    
    bool IsVisible(Camera camera, float4x4 trans)
    {
        float sx = length(float3(trans[0][0], trans[0][1], trans[0][2]));
        float sy = length(float3(trans[1][0], trans[1][1], trans[1][2]));
        float sz = length(float3(trans[2][0], trans[2][1], trans[2][2]));
        
        float3 radius = bound.radius * float3(sx, sy, sz);
        float3 center = mul(trans, float4(bound.center, 1.0)).xyz;
        
        // Frustum Culling
        for (uint i = 0; i < 6; i++)
        {
            if (dot(camera.frustum[i], float4(center, 1)) + length(radius) < 0.0)
            {
                return false;
            }
        }

        // Cone Culling
        cone_apex = mul(trans, float4(cone_apex, 1.0)).xyz;
        
        float3x3 rotation = float3x3(
            trans[0].xyz / sx,
            trans[1].xyz / sy,
            trans[2].xyz / sz
        );

        cone_axis = mul(rotation, cone_axis);
        return dot(normalize(camera.position - cone_apex), cone_axis) < cone_cutoff;;
    }
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
struct MeshDrawCommand
{
    uint drawID;
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
    uint taskCount;
    uint firstTask;
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

#define Material_Matte 0
#define Material_Plastic 1
#define Material_Metal 2
#define Material_Mirror 3
#define Material_Substrate 4
#define Material_Glass 5
#define Material_Disney 6

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

struct ShadowPayload
{
    bool visibility;
};

struct Interaction
{
    float3 position;
    float3 normal;
    float3 ffnormal;
    float3 tangent;
    float3 bitangent;
    float2 texCoord;
    float3 wo;
    
    void CreateCoordinateSystem()
    {
        const float3 ref = abs(dot(ffnormal, float3(0, 1, 0))) > 0.99f ? float3(0, 0, 1) : float3(0, 1, 0);

        tangent = normalize(cross(ref, ffnormal));
        bitangent = cross(ffnormal, tangent);
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
    
    bool IsSurfaceInteraction()
    {
        return ffnormal.x != 0.0 || ffnormal.y != 0.0 || ffnormal.z != 0.0;
    }
};

float3 OffsetRay(float3 p, float3 n)
{
    const float intScale = 256.0f;
    const float floatScale = 1.0f / 65536.0f;
    const float origin = 1.0f / 32.0f;

    int3 of_i = int3(intScale * n.x, intScale * n.y, intScale * n.z);

    float3 p_i = float3(asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x, //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y, //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}

RayDesc SpawnRay(Interaction isect, float3 wi)
{
    RayDesc ray;
    ray.Direction = wi;
    ray.Origin = OffsetRay(isect.position, dot(wi, isect.ffnormal) > 0.0 ? isect.ffnormal : -isect.ffnormal);
    return ray;
}

#endif