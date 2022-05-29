RWStructuredBuffer<uint> morton_codes_buffer : register(u0);
RWStructuredBuffer<uint> indices_buffer : register(u1);

#ifndef GROUP_SIZE
#define GROUP_SIZE 1024
#endif

[[vk::push_constant]]
struct
{
    uint h;
    uint size;
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
    uint GroupIndex : SV_GroupIndex;
};

groupshared uint local_morton_codes_cache[GROUP_SIZE * 2];
groupshared uint local_indices_cache[GROUP_SIZE * 2];

void GlobalCompareAndSwap(uint2 idx)
{
    if (morton_codes_buffer[idx.x] > morton_codes_buffer[idx.y])
    {
        uint tmp = morton_codes_buffer[idx.x];
        morton_codes_buffer[idx.x] = morton_codes_buffer[idx.y];
        morton_codes_buffer[idx.y] = tmp;
        
        tmp = indices_buffer[idx.x];
        indices_buffer[idx.x] = indices_buffer[idx.y];
        indices_buffer[idx.y] = tmp;
    }
}

void GlobalFlip(uint h, CSParam param)
{
    if (GROUP_SIZE * 2 > h)
    {
        return;
    }
    
    uint t = param.DispatchThreadID.x;
    uint half_h = h >> 1;
    uint2 indices = h * ((2 * t) / h) + uint2(t % half_h, h - 1 - (t % half_h));
    
    if (indices.x < push_constants.size && indices.y < push_constants.size)
    {
        GlobalCompareAndSwap(indices);
    }
}

void GlobalDisperse(uint h, CSParam param)
{
    if (GROUP_SIZE * 2 > h)
    {
        return;
    }
    
    uint t = param.DispatchThreadID.x;
    uint half_h = h >> 1;
    uint2 indices = h * ((2 * t) / h) + uint2(t % half_h, half_h + (t % half_h));
    
    if (indices.x < push_constants.size && indices.y < push_constants.size)
    {
        GlobalCompareAndSwap(indices);
    }
}

void LocalCompareAndSwap(uint2 idx)
{
    if (local_morton_codes_cache[idx.x] > local_morton_codes_cache[idx.y])
    {
        uint local_morton = local_morton_codes_cache[idx.x];
        local_morton_codes_cache[idx.x] = local_morton_codes_cache[idx.y];
        local_morton_codes_cache[idx.y] = local_morton;
        
        uint local_indices = local_indices_cache[idx.x];
        local_indices_cache[idx.x] = local_indices_cache[idx.y];
        local_indices_cache[idx.y] = local_indices;
    }
}

void LocalFlip(uint h, CSParam param)
{
    uint t = param.GroupThreadID.x;
    uint half_h = h >> 1;
    uint2 indices = h * ((2 * t) / h) + uint2(t % half_h, h - 1 - (t % half_h));
    if (param.GroupID.x * GROUP_SIZE + indices.x < push_constants.size &&
       param.GroupID.x * GROUP_SIZE + indices.y < push_constants.size)
    {
        LocalCompareAndSwap(indices);
    }
}

void LocalDisperse(uint h, CSParam param)
{
    uint t = param.GroupThreadID.x;
    uint half_h = h >> 1;
    uint2 indices = h * ((2 * t) / h) + uint2(t % half_h, half_h + (t % half_h));
    if (param.GroupID.x * GROUP_SIZE + indices.x < push_constants.size &&
        param.GroupID.x * GROUP_SIZE + indices.y < push_constants.size)
    {
        LocalCompareAndSwap(indices);
    }
}

void LocalBMs(uint h, CSParam param)
{
   [unroll]
    for (uint h1 = 2; h1 <= h; h1 <<= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        LocalFlip(h1, param);
        
        [unroll]
        for (uint h2 = h1 >> 1; h2 > 1; h2 >>= 1)
        {
            GroupMemoryBarrierWithGroupSync ();
            LocalDisperse(h2, param);
        }
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(CSParam param)
{
    uint offset = GROUP_SIZE * 2 * param.GroupID.x;
    uint t = param.GroupThreadID.x;
    
#ifdef LOCAL_BMS
    local_morton_codes_cache[t * 2] = morton_codes_buffer[offset + t * 2];
    local_morton_codes_cache[t * 2 + 1] = morton_codes_buffer[offset + t * 2 + 1];
    local_indices_cache[t * 2] = indices_buffer[offset + t * 2];
    local_indices_cache[t * 2 + 1] = indices_buffer[offset + t * 2 + 1];
    LocalBMs(push_constants.h, param);
     GroupMemoryBarrierWithGroupSync ();
    morton_codes_buffer[offset + t * 2] = local_morton_codes_cache[t * 2];
    morton_codes_buffer[offset + t * 2 + 1] = local_morton_codes_cache[t * 2 + 1];
    indices_buffer[offset + t * 2] = local_indices_cache[t * 2];
    indices_buffer[offset + t * 2 + 1] = local_indices_cache[t * 2 + 1];
#endif
    
#ifdef LOCAL_DISPERSE
    local_morton_codes_cache[t * 2] = morton_codes_buffer[offset + t * 2];
    local_morton_codes_cache[t * 2 + 1] = morton_codes_buffer[offset + t * 2 + 1];
    local_indices_cache[t * 2] = indices_buffer[offset + t * 2];
    local_indices_cache[t * 2 + 1] = indices_buffer[offset + t * 2 + 1];
    for (uint h = push_constants.h; h > 1; h >>= 1)
    {
        GroupMemoryBarrierWithGroupSync ();
        LocalDisperse(h, param);
    }
    GroupMemoryBarrierWithGroupSync ();
    morton_codes_buffer[offset + t * 2] = local_morton_codes_cache[t * 2];
    morton_codes_buffer[offset + t * 2 + 1] = local_morton_codes_cache[t * 2 + 1];
    indices_buffer[offset + t * 2] = local_indices_cache[t * 2];
    indices_buffer[offset + t * 2 + 1] = local_indices_cache[t * 2 + 1];
#endif
    
#ifdef GLOBAL_FLIP
    GlobalFlip(push_constants.h, param);
#endif
    
#ifdef GLOBAL_DISPERSE
    GlobalDisperse(push_constants.h, param);
#endif
    
}