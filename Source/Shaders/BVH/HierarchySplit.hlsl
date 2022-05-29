#include "../ShaderInterop.hpp"

StructuredBuffer<uint> morton_codes_buffer : register(t0);
RWStructuredBuffer<HierarchyNode> hierarchy_buffer : register(u1);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
    uint GroupIndex : SV_GroupIndex;
};

[[vk::push_constant]]
struct
{
    uint leaf_count;
    uint node_count;
} push_constants;

int GetLongestCommonPerfix(uint lhs_idx, uint rhs_idx)
{
    if (lhs_idx >= push_constants.leaf_count || rhs_idx >= push_constants.leaf_count)
    {
        return -1;
    }
    else
    {
        uint morton_code_lhs = morton_codes_buffer[lhs_idx];
        uint morton_code_rhs = morton_codes_buffer[rhs_idx];
        if (morton_code_lhs != morton_code_rhs)
        {
            return 31 - firstbithigh(morton_code_lhs ^ morton_code_rhs);
        }
        else
        {
            return 32 + 31 - firstbithigh(morton_code_lhs ^ morton_code_rhs);
        }
    }
}

void WriteChildren(uint child, uint parent)
{
    hierarchy_buffer[child].parent = parent;
}

void WriteParent(uint parent, uint lchild, uint rchild)
{
    hierarchy_buffer[parent].left_child = lchild;
    hierarchy_buffer[parent].right_child = rchild;
}

uint2 DetermineRange(uint idx)
{
    int d = GetLongestCommonPerfix(idx, idx + 1) - GetLongestCommonPerfix(idx, idx - 1);
    d = clamp(d, -1, 1);
    int min_prefix = GetLongestCommonPerfix(idx, idx - d);
    int max_length = 2;
    while (GetLongestCommonPerfix(idx, idx + max_length * d) > min_prefix)
    {
        max_length *= 2;
    }
    int length = 0;
    for (int t = max_length / 2; t >= 1; t /= 2)
    {
        if (GetLongestCommonPerfix(idx, idx + (length + t) * d) > min_prefix)
        {
            length += t;
        }
    }
    int j = idx + length * d;
    return uint2(min(idx, j), max(idx, j));
}

uint FindSplit(uint start, uint end)
{
    int common_prefix = GetLongestCommonPerfix(start, end);
    uint split = start;
    int step = end - start;
    
    do
    {
        step = (step + 1) >> 1;
        int new_split = split + step;
        if (new_split < end)
        {
            int split_prefix = GetLongestCommonPerfix(start, new_split);
            if (split_prefix > common_prefix)
            {
                split = new_split;
            }
        }
    } while (step > 1);
    
    return split;
}

void GenerateHierarchy(uint idx)
{
    uint2 range = DetermineRange(idx);
    uint start = range.x;
    uint end = range.y;
    

    
    uint split = FindSplit(start, end);
    uint leaf_offset = push_constants.leaf_count - 1;
    uint left_child = split;
    uint right_child = split + 1;
    if (split == start)
    {
        left_child += leaf_offset;
    }
    if (split + 1 == end)
    {
        right_child += leaf_offset;
    }
    
    WriteChildren(left_child, idx);
    WriteChildren(right_child, idx);
    WriteParent(idx, left_child, right_child);
}

[numthreads(1024, 1, 1)]
void main(CSParam param)
{
    if (param.DispatchThreadID.x < push_constants.node_count)
    {
        hierarchy_buffer[param.DispatchThreadID.x].parent = ~0U;
        hierarchy_buffer[param.DispatchThreadID.x].left_child = ~0U;
        hierarchy_buffer[param.DispatchThreadID.x].right_child = ~0U;
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (param.DispatchThreadID.x >= push_constants.leaf_count - 1)
    {
        return;
    }
    
    GenerateHierarchy(param.DispatchThreadID.x);
}