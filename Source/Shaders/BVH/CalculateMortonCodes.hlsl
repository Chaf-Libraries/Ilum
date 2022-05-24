#include "../ShaderInterop.hpp"

#define BUILD_BLAS

//ConstantBuffer<SceneInfo> scene_info : register(b0);
RWStructuredBuffer<uint> morton_codes_buffer : register(t1);
RWStructuredBuffer<uint> indices_buffer : register(t2);

#ifdef BUILD_BLAS
StructuredBuffer<Vertex> vertices[] : register(t3);
StructuredBuffer<uint> indices[] : register(t4);
[[vk::push_constant]]
struct
{
    float3 aabb_min;
    uint primitive_count;
    float3 aabb_max;
    uint instance_id;
    uint index_offset;
} push_constants;
#endif

#ifdef BUILD_TLAS
ConstantBuffer<Instance> instances[] : register(b3);
#endif

struct CSParam
{
    uint DispatchThreadID : SV_DispatchThreadID;
};

uint ExpandBits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint MortonCodeFromUnitCoord(float3 unit_coord)
{
    float x = min(max(x * 1024.0, 0.0), 1023.0);
    float y = min(max(y * 1024.0, 0.0), 1023.0);
    float z = min(max(z * 1024.0, 0.0), 1023.0);
    
    uint xx = ExpandBits(asuint(x));
    uint yy = ExpandBits(asuint(y));
    uint zz = ExpandBits(asuint(z));
    return xx * 4 + yy * 2 + zz;
}

uint MortonCode(float3 elem_centroid)
{
    const float epsilon = 0.00001;
    
    const float3 aabb_min = push_constants.aabb_min;
    const float3 aabb_max = push_constants.aabb_max;
    float3 scene_dimension = max(aabb_max - aabb_min, epsilon);
    float3 unit_coord = (elem_centroid - aabb_min) / scene_dimension;
    
    return MortonCodeFromUnitCoord(unit_coord);
}

float3 GetCentroid(float elem_index)
{
#ifdef BUILD_TLAS
    const float3 aabb_min = instances[elem_index].aabb_min;
    const float3 aabb_max = instances[elem_index].aabb_max;
    return (aabb_min + aabb_max) * 0.5;
#endif
    
#ifdef BUILD_BLAS
    float3 v1 = vertices[push_constants.instance_id][indices[push_constants.instance_id][elem_index * 3]].position;
    float3 v2 = vertices[push_constants.instance_id][indices[push_constants.instance_id][elem_index * 3 + 1]].position;
    float3 v3 = vertices[push_constants.instance_id][indices[push_constants.instance_id][elem_index * 3 + 2]].position;
    return (v1 + v2 + v3) / 3.0;
#endif
}

[numthreads(32, 1, 1)]
void main(CSParam param)
{
    uint elem_index = 0;
#ifdef BUILD_TLAS
    if (scene_info.instance_count <= param.DispatchThreadID.x)
    {
        return;
    }
    elem_index = param.DispatchThreadID.x;
#endif
    
#ifdef BUILD_BLAS
    if (push_constants.primitive_count <= param.DispatchThreadID.x)
    {
        return;
    }
    elem_index = push_constants.index_offset + param.DispatchThreadID.x;
#endif
    
    float3 elem_centroid = GetCentroid(param.DispatchThreadID.x);
    uint morton_code = MortonCode(elem_centroid);
    morton_codes_buffer[elem_index] = morton_code;
}