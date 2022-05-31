#include "../ShaderInterop.hpp"
#include "../RayTrace.hlsli"

ConstantBuffer<Camera> camera : register(b0);
StructuredBuffer<HierarchyNode> blas_hierarchy[] : register(t1);
StructuredBuffer<AABB> blas_aabb[] : register(t2);
ConstantBuffer<Instance> instances[] : register(b3);
RWTexture2D<float4> result : register(u4);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
    uint GroupIndex : SV_GroupIndex;
};

uint hash(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

bool BVHTreeTraversal(RayDesc ray, out float depth)
{
    uint node = 0;
    float min_t = 1e32;
    float t = 0.0;
    float current_depth = 0.0;
    bool find_leaf = false;
    float max_depth = 0.0;
    bool terminal = false;
    
    while (true)
    {
        max_depth = max(max_depth, current_depth);
        
        if (Intersection(blas_aabb[0][node].Transform(instances[0].transform), ray, t))
        {
            if (blas_hierarchy[0][node].left_child == ~0U &&
                blas_hierarchy[0][node].right_child == ~0U)
            {
                if (min_t > t)
                {
                    min_t = t;
                    depth = current_depth;
                    find_leaf = true;
                }
                
                node = blas_hierarchy[0][node].parent;
                current_depth -= 1.0;
            }
            else
            {
                node = blas_hierarchy[0][node].left_child;
                current_depth += 1.0;
                continue;
            }
        }
        
        while (true)
        {
            if (blas_hierarchy[0][node].right_sibling != ~0U)
            {
                node = blas_hierarchy[0][node].right_sibling;
                break;
            }
            node = blas_hierarchy[0][node].parent;
            current_depth -= 1.0;
            if (node == ~0U)
            {
                terminal = true;
                break;
            }
        }
        
        if (terminal)
        {
            break;
        }
    }
    
    if(!find_leaf)
    {
        depth = max_depth;
        return false;
    }
    return true;
}

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
    screen_coords.y = 1.0 - screen_coords.y;
    screen_coords = screen_coords * 2.0 - 1.0;
    
    RayDesc ray = camera.CastRay(screen_coords);
    
    float t = 0;
    float depth = 0.0;
    
    // if (!BVHTreeTraversal(ray, depth))
    // {
    //     result[param.DispatchThreadID.xy] = 0.0;
    //     return;
    // }
    
    BVHTreeTraversal(ray, depth);
        
    uint item_count = 0;
    uint item_stride = 0;
    blas_hierarchy[0].GetDimensions(item_count, item_stride);
    float max_depth = log2(item_count) * 2;
    
    uint mhash = hash(asuint(t));
    float3 color = depth / max_depth;
    result[param.DispatchThreadID.xy] = float4(color, 1.0);
}