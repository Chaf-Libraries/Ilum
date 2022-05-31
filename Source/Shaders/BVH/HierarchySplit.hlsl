#include "../ShaderInterop.hpp"

StructuredBuffer<uint> morton_codes_buffer : register(t0);
RWStructuredBuffer<BVHNode> bvh_buffer : register(u1);
StructuredBuffer<uint> indices_buffer : register(t2);

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
} push_constants;

int CountLeadingZeroes(uint num)
{
    return 31 - firstbithigh(num);
}

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
            return CountLeadingZeroes(morton_codes_buffer[lhs_idx] ^ morton_codes_buffer[rhs_idx]);
        }
        else
        {
            return CountLeadingZeroes(lhs_idx ^ rhs_idx) + 31;
        }
    }
}

void WriteChildren(uint child, uint parent)
{
    bvh_buffer[child].parent = parent;
}

void WriteParent(uint parent, uint lchild, uint rchild)
{
    bvh_buffer[parent].left_child = lchild;
    bvh_buffer[parent].right_child = rchild;
}

/*https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees*/
void GenerateHierarchy(int idx)
{
    int d = clamp(GetLongestCommonPerfix(idx, idx + 1) - GetLongestCommonPerfix(idx, idx - 1), -1, 1);
    int min_prefix = GetLongestCommonPerfix(idx, idx - d);
    int l_max = 2;
    while (GetLongestCommonPerfix(idx, idx + l_max * d) > min_prefix)
    {
        l_max = l_max * 2;
    }
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2)
    {
        if (GetLongestCommonPerfix(idx, idx + (l + t) * d) > min_prefix)
        {
            l = l + t;
        }
    }
    int j = idx + l * d;
    int node_prefix = GetLongestCommonPerfix(idx, j);
    int s = 0;
    float n = 2;
    while (true)
    {
        t = (int) (ceil((float) l / n));
        if (GetLongestCommonPerfix(idx, idx + (s + t) * d) > node_prefix)
        {
            s = s + t;
        }
        n *= 2.0;
        if (t == 1)
        {
            break;
        }
    }
    
    int leaf_offset = push_constants.leaf_count - 1;
    int gamma = idx + s * d + min(d, 0);
    uint left_child = gamma;
    uint right_child = gamma + 1;
    
    if (min(idx, j) == gamma)
    {
        left_child += leaf_offset;
    }
    if (max(idx, j) == gamma + 1)
    {
        right_child += leaf_offset;
    }
    
    WriteParent(idx, left_child, right_child);
    WriteChildren(left_child, idx);
    WriteChildren(right_child, idx);
}

[numthreads(1024, 1, 1)]
void main(CSParam param)
{
#ifdef INITIALIZE
    if (param.DispatchThreadID.x < push_constants.leaf_count * 2 - 1)
    {
        bvh_buffer[param.DispatchThreadID.x].parent = ~0U;
        bvh_buffer[param.DispatchThreadID.x].left_child = ~0U;
        bvh_buffer[param.DispatchThreadID.x].right_child = ~0U;
        
        if (param.DispatchThreadID.x >= push_constants.leaf_count - 1)
        {
            bvh_buffer[param.DispatchThreadID.x].prim_id = indices_buffer[param.DispatchThreadID.x + 1 - push_constants.leaf_count];
        }
        else
        {
            bvh_buffer[param.DispatchThreadID.x].prim_id = ~0U;
        }
    }
#endif
    
#ifdef SPLIT
    if (param.DispatchThreadID.x < push_constants.leaf_count - 1)
    {
        GenerateHierarchy(param.DispatchThreadID.x);
    }
#endif
}