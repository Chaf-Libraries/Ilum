#ifndef COMMON_HLSLI
#define COMMON_HLSLI

#include "Math.hlsli"

#define MAX_BONE_INFLUENCE 8

#define MESH_TYPE 0
#define SKINNED_MESH_TYPE 1

struct DrawIndirectCommand
{
    uint vertex_count;
    uint instance_count;
    uint vertex_offset;
    uint instance_offset;
};

struct DrawIndexedIndirectCommand
{
    uint index_count;
    uint instance_count;
    uint index_offset;
    int vertex_offset;
    uint instance_offset;
};

struct DispatchIndirectCommand
{
    uint x;
    uint y;
    uint z;
};

struct DrawMeshTasksIndirectCommand
{
    uint group_count_x;
    uint group_count_y;
    uint group_count_z;
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

    uint data_offset;
    uint vertex_offset;
    uint vertex_count;
    uint triangle_count;
    
    float3 cone_axis;
    float cone_cutoff;
    float3 cone_apex;
    
    uint visible;
};

struct Instance
{
    float4x4 transform;

    uint mesh_id;
    uint material_id;
    uint animation_id;
    uint visible;
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

bool IsInsideFrustum(Meshlet meshlet, float4x4 model, float4 frustum[6], float3 eye)
{
    float sx = length(float3(model[0][0], model[0][1], model[0][2]));
    float sy = length(float3(model[1][0], model[1][1], model[1][2]));
    float sz = length(float3(model[2][0], model[2][1], model[2][2]));
        
    float3 radius = meshlet.radius * float3(sx, sy, sz);
    float3 center = mul(model, float4(meshlet.center, 1.0)).xyz;
        
        // Frustum Culling
    for (uint i = 0; i < 6; i++)
    {
        if (dot(frustum[i], float4(center, 1)) + length(radius) < 0.0)
        {
            return false;
        }
    }
        
    // Cone Culling
    float3 cone_apex = mul(model, float4(meshlet.cone_apex, 1.0)).xyz;
        
    float3x3 rotation = float3x3(
            model[0].xyz / sx,
            model[1].xyz / sy,
            model[2].xyz / sz
        );

    float3 cone_axis = mul(rotation, meshlet.cone_axis);
    float result = dot(normalize(mul(model, float4(meshlet.cone_apex, 1.0)).xyz - eye), cone_axis);
    return result < meshlet.cone_cutoff || result > 0.f;
}

bool IsInsideFrustum(Meshlet meshlet, float4x4 model, float4 frustum[6], float3 eye, float3 offset)
{
    float sx = length(float3(model[0][0], model[0][1], model[0][2]));
    float sy = length(float3(model[1][0], model[1][1], model[1][2]));
    float sz = length(float3(model[2][0], model[2][1], model[2][2]));
        
    float3 radius = meshlet.radius * float3(sx, sy, sz);
    float3 center = mul(model, float4(meshlet.center, 1.0)).xyz - offset;
        
        // Frustum Culling
    for (uint i = 0; i < 6; i++)
    {
        if (dot(frustum[i], float4(center, 1)) + length(radius) < 0.0)
        {
            return false;
        }
    }
        
    // Cone Culling
    float3 cone_apex = mul(model, float4(meshlet.cone_apex, 1.0)).xyz - offset;
        
    float3x3 rotation = float3x3(
            model[0].xyz / sx,
            model[1].xyz / sy,
            model[2].xyz / sz
        );

    float3 cone_axis = mul(rotation, meshlet.cone_axis);
    float result = dot(normalize(mul(model, float4(meshlet.cone_apex, 1.0)).xyz - eye) - offset, cone_axis);
    return result < meshlet.cone_cutoff || result > 0.f;
}

struct View
{
    float4 frustum[6];
    float4x4 view_matrix;
    float4x4 inv_view_matrix;
    float4x4 projection_matrix;
    float4x4 inv_projection_matrix;
    float4x4 view_projection_matrix;
    float4x4 inv_view_projection_matrix;
    float3 position;
    uint frame_count;
    float2 viewport;
    
    bool IsVisible(Meshlet meshlet, float4x4 model)
    {
        return IsInsideFrustum(meshlet, model, frustum, position);
    }
    
    void CastRay(float2 scene_uv, float2 frame_dim, out RayDesc ray, out RayDiff ray_diff)
    {
        float yaw = atan2(-view_matrix[2][2], -view_matrix[0][2]);
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

uint PackVisibilityBuffer(uint mesh_type, uint instance_id, uint primitive_id)
{
    // Mesh Type 1
    // Instance ID 8
    // Primitive ID 23
    uint vbuffer = 0;
    vbuffer += mesh_type & 0x1;
    vbuffer += (instance_id & 0xff) << 1;
    vbuffer += (primitive_id & 0x7fffff) << 9;
    return vbuffer;
}

void UnPackVisibilityBuffer(uint visibility_buffer, out uint mesh_type, out uint instance_id, out uint primitive_id)
{
    // Mesh Type 1
    // Instance ID 8
    // Primitive ID 23
    mesh_type = visibility_buffer & 0x1;
    instance_id = (visibility_buffer >> 1) & 0xff;
    primitive_id = (visibility_buffer >> 9) & 0x7fffff;
}

uint PackXY(uint x, uint y)
{
    return (x & 0xffff) + ((y & 0xffff) << 16);
}

void UnpackXY(uint xy, out uint x, out uint y)
{
    x = xy & 0xffff;
    y = (xy >> 16) & 0xffff;
}

float Luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f)); //color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
}

void CalculateFrustum(float4x4 view_projection, out float4 frustum[6])
{
    view_projection = transpose(view_projection);
    
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
        float len = length(frustum[i].xyz);
        frustum[i] /= len;
    }
}

#endif