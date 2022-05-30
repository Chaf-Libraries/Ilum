#include "../ShaderInterop.hpp"
#include "../RayTrace.hlsli"

ConstantBuffer<Camera> camera : register(b0);
StructuredBuffer<HierarchyNode> blas_hierarchy[] : register(t1);
StructuredBuffer<AABB> blas_aabb[] : register(t2);
RWTexture2D<float4> result : register(u3);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
    uint GroupIndex : SV_GroupIndex;
};

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    uint2 extent;
    result.GetDimensions(extent.x, extent.y);

    if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
    {
        return;
    }
    
    float2 screen_coords = (float2(param.DispatchThreadID.xy) + 0.5) / float2(extent);
    
    RayDesc ray = camera.CastRay(screen_coords);
    
    uint node = 0;
    float t = 0;
    float depth = 1.0;
    
    if (Intersection(blas_aabb[0][node], ray, t))
    {
        result[param.DispatchThreadID.xy] = 0.3;
        return;
    }
    else
    {
        result[param.DispatchThreadID.xy] = 1.;
        return;
    }
    
    while (node != ~0U)
    {
        uint left_child = blas_hierarchy[0][node].left_child;
        uint right_child = blas_hierarchy[0][node].right_child;
        
        if (Intersection(blas_aabb[0][left_child], ray, t))
        {
            depth += 1.0;
            node = left_child;
        }
        else if (Intersection(blas_aabb[0][right_child], ray, t))
        {
            depth += 1.0;
            node = right_child;
        }
        else
        {
            break;
        }
    }
    
    result[param.DispatchThreadID.xy] = 1.0 / depth;
}