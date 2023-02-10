#ifndef __CULLING_HLSL__
#define __CULLING_HLSL__
#include "ShaderInterop.hpp"

struct Frustum
{
    float4 planes[6];
    
    void Build(Camera camera)
    {
        float4x4 view_projection_transpose = transpose(camera.view_projection);
        
        // Left
        planes[0].x = view_projection_transpose[0].w + view_projection_transpose[0].x;
        planes[0].y = view_projection_transpose[1].w + view_projection_transpose[1].x;
        planes[0].z = view_projection_transpose[2].w + view_projection_transpose[2].x;
        planes[0].w = view_projection_transpose[3].w + view_projection_transpose[3].x;

	    // Right
        planes[1].x = view_projection_transpose[0].w - view_projection_transpose[0].x;
        planes[1].y = view_projection_transpose[1].w - view_projection_transpose[1].x;
        planes[1].z = view_projection_transpose[2].w - view_projection_transpose[2].x;
        planes[1].w = view_projection_transpose[3].w - view_projection_transpose[3].x;

	    // Top
        planes[2].x = view_projection_transpose[0].w - view_projection_transpose[0].y;
        planes[2].y = view_projection_transpose[1].w - view_projection_transpose[1].y;
        planes[2].z = view_projection_transpose[2].w - view_projection_transpose[2].y;
        planes[2].w = view_projection_transpose[3].w - view_projection_transpose[3].y;

	    // Bottom
        planes[3].x = view_projection_transpose[0].w + view_projection_transpose[0].y;
        planes[3].y = view_projection_transpose[1].w + view_projection_transpose[1].y;
        planes[3].z = view_projection_transpose[2].w + view_projection_transpose[2].y;
        planes[3].w = view_projection_transpose[3].w + view_projection_transpose[3].y;

	    // Near
        planes[4].x = view_projection_transpose[0].w + view_projection_transpose[0].z;
        planes[4].y = view_projection_transpose[1].w + view_projection_transpose[1].z;
        planes[4].z = view_projection_transpose[2].w + view_projection_transpose[2].z;
        planes[4].w = view_projection_transpose[3].w + view_projection_transpose[3].z;

	    // Far
        planes[5].x = view_projection_transpose[0].w - view_projection_transpose[0].z;
        planes[5].y = view_projection_transpose[1].w - view_projection_transpose[1].z;
        planes[5].z = view_projection_transpose[2].w - view_projection_transpose[2].z;
        planes[5].w = view_projection_transpose[3].w - view_projection_transpose[3].z;
        
        for (uint i = 0; i < 6; i++)
        {
            planes[i] /= length(planes[i].xyz);
        }
    }
};

bool IsMeshletVisible(Meshlet meshlet, float4x4 trans, Camera camera)
{
    float sx = length(float3(trans[0][0], trans[0][1], trans[0][2]));
    float sy = length(float3(trans[1][0], trans[1][1], trans[1][2]));
    float sz = length(float3(trans[2][0], trans[2][1], trans[2][2]));
        
    float3 radius = meshlet.bound.radius * float3(sx, sy, sz);
    float3 center = mul(trans, float4(meshlet.bound.center, 1.0)).xyz;
    
    Frustum frustum;
    frustum.Build(camera);
        
        // Frustum Culling
    for (uint i = 0; i < 6; i++)
    {
        if (dot(frustum.planes[i], float4(center, 1)) + length(radius) < 0.0)
        {
            return false;
        }
    }
        
    // Cone Culling
    float3 cone_axis = normalize(mul((float3x3) trans, meshlet.bound.cone_axis));
    if (dot(camera.position - center.xyz, cone_axis) >= meshlet.bound.cone_cut_off * length(center.xyz - camera.position) + length(radius))
    {
        return false;
    }
    return true;
}

#endif