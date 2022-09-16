#include "../ShaderInterop.hpp"
#include "../Constants.hlsli"

RWStructuredBuffer<BVHNode> bvh_buffer : register(u0);
RWStructuredBuffer<uint> hierarchy_flags : register(u1);

#ifdef BUILD_BLAS
StructuredBuffer<Vertex> vertices : register(t2);
StructuredBuffer<uint> indices : register(t3);
#endif

#ifdef BUILD_TLAS
ConstantBuffer<Instance> instances[] : register(b2);
#endif

[[vk::push_constant]]
struct
{
    uint leaf_count;
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
    uint GroupIndex : SV_GroupIndex;
};

[numthreads(1024, 1, 1)]
void main(CSParam param)
{
#ifdef INITIALIZE
    if (param.DispatchThreadID.x < push_constants.leaf_count * 2 -1)
    {
        bvh_buffer[param.DispatchThreadID.x].aabb.max_val = -Infinity;
        bvh_buffer[param.DispatchThreadID.x].aabb.min_val = Infinity;
        hierarchy_flags[param.DispatchThreadID.x] = 0;
    }
#endif
    
#ifdef GENERATION
    if (param.DispatchThreadID.x >= push_constants.leaf_count)
    {
        return;
    }
    
    // Build leaf AABB
    uint leaf = push_constants.leaf_count - 1 + param.DispatchThreadID.x;
    uint primitive_idx = bvh_buffer[leaf].prim_id;
    
#ifdef BUILD_BLAS
    float3 v1 = vertices[indices[primitive_idx * 3]].position.xyz;
    float3 v2 = vertices[indices[primitive_idx * 3 + 1]].position.xyz;
    float3 v3 = vertices[indices[primitive_idx * 3 + 2]].position.xyz;
    bvh_buffer[leaf].aabb.min_val = float4(min(min(v1, v2), v3), 0.0);
    bvh_buffer[leaf].aabb.max_val = float4(max(max(v1, v2), v3), 0.0);
#endif
    
#ifdef BUILD_TLAS
    bvh_buffer[leaf].aabb.min_val = float4(instances[primitive_idx].aabb_min, 0.0);
    bvh_buffer[leaf].aabb.max_val = float4(instances[primitive_idx].aabb_max, 0.0);
#endif
    
    hierarchy_flags[leaf] = 1;
    
    // Build node AABB
    uint parent = bvh_buffer[leaf].parent;
    while (parent != ~0U)
    {
        uint prev_flag = 0;
        InterlockedAdd(hierarchy_flags[parent], 1, prev_flag);
        DeviceMemoryBarrier();
                        
        if (prev_flag != 1)
        {
            return;
        }
        
        uint left_child = bvh_buffer[parent].left_child;
        uint right_child = bvh_buffer[parent].right_child;

        AABB left_aabb = bvh_buffer[left_child].aabb;
        AABB right_aabb = bvh_buffer[right_child].aabb;
        
        bvh_buffer[parent].aabb.min_val = min(left_aabb.min_val, right_aabb.min_val);
        bvh_buffer[parent].aabb.max_val = max(left_aabb.max_val, right_aabb.max_val);
        
        parent = bvh_buffer[parent].parent;
    }
#endif
}