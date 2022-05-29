#include "../ShaderInterop.hpp"
#include "../Constants.hlsli"

StructuredBuffer<uint> primitive_indices_buffer : register(t0);
StructuredBuffer<Vertex> vertices : register(t1);
StructuredBuffer<uint> indices : register(t2);
StructuredBuffer<HierarchyNode> hierarchy_buffer : register(t3);
RWStructuredBuffer<AABB> aabbs_buffer : register(u4);
RWStructuredBuffer<uint> hierarchy_flags : register(u5);

[[vk::push_constant]]
struct
{
    uint leaf_count;
    uint node_count;
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
    if (param.DispatchThreadID.x < push_constants.node_count)
    {
        aabbs_buffer[param.DispatchThreadID.x].max_val = -Infinity;
        aabbs_buffer[param.DispatchThreadID.x].min_val = Infinity;
        hierarchy_flags[param.DispatchThreadID.x] = 0;
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (param.DispatchThreadID.x >= push_constants.leaf_count)
    {
        return;
    }
    
    // Build leaf AABB
    uint leaf = push_constants.leaf_count - 1 + param.DispatchThreadID.x;
    uint primitive_idx = primitive_indices_buffer[param.DispatchThreadID.x];
    float3 v1 = vertices[indices[primitive_idx * 3]].position.xyz;
    float3 v2 = vertices[indices[primitive_idx * 3 + 1]].position.xyz;
    float3 v3 = vertices[indices[primitive_idx * 3 + 2]].position.xyz;
    aabbs_buffer[leaf].min_val = float4(min(min(v1, v2), v3), 0.0);
    aabbs_buffer[leaf].max_val = float4(max(max(v1, v2), v3), 0.0);
    hierarchy_flags[leaf] = 1;
    GroupMemoryBarrierWithGroupSync();
    
    // Build node AABB
    uint parent = hierarchy_buffer[leaf].parent;
    while (parent != ~0U)
    {
        uint prev_flag = 0;
        InterlockedAdd(hierarchy_flags[parent], 1, prev_flag);
                
        if (prev_flag != 1)
        {
            return;
        }
        
        uint left_child = hierarchy_buffer[parent].left_child;
        uint right_child = hierarchy_buffer[parent].right_child;

        AABB left_aabb = aabbs_buffer[left_child];
        AABB right_aabb = aabbs_buffer[right_child];
        
        aabbs_buffer[parent].min_val = min(left_aabb.min_val, right_aabb.min_val);
        aabbs_buffer[parent].max_val = max(left_aabb.max_val, right_aabb.max_val);
        
        parent = hierarchy_buffer[parent].parent;
    
        GroupMemoryBarrierWithGroupSync();
    }

}