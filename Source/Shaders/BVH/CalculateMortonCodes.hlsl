#include "../ShaderInterop.hpp"

#define BUILD_BLAS

//ConstantBuffer<SceneInfo> scene_info : register(b0);
RWStructuredBuffer<uint> morton_codes_buffer : register(u0);
RWStructuredBuffer<uint> indices_buffer : register(u1);

#ifdef BUILD_BLAS
StructuredBuffer<Vertex> vertices : register(t2);
StructuredBuffer<uint> indices : register(t3);
[[vk::push_constant]]
struct
{
    float3 aabb_min;
    uint primitive_count;
    float3 aabb_max;
} push_constants;
#endif

#ifdef BUILD_TLAS
ConstantBuffer<Instance> instances[] : register(b3);
#endif

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

#define BIT(x) (1U << (x))

uint MortonCodeFromUnitCoord(float3 unit_coord)
{
    uint morton_code = 0;
    const uint num_bits = 10;
    float max_coord = pow(2, num_bits);

    float3 adjusted_coord = min(max(unit_coord * max_coord, 0.0), max_coord);
    const uint num_axis = 3;
    uint coords[num_axis] = { adjusted_coord.y, adjusted_coord.x, adjusted_coord.z };
    for (uint bit_index = 0; bit_index < num_bits; bit_index++)
    {
        for (uint axis = 0; axis < num_axis; axis++)
        {
            uint bit = BIT(bit_index) & coords[axis];
            if (bit)
            {
                morton_code |= BIT(bit_index * num_axis + axis);
            }
        }
    }
    return morton_code;
}

[numthreads(32, 1, 1)]
void main(CSParam param)
{
    if (push_constants.primitive_count <= param.DispatchThreadID.x)
    {
        return;
    }

    
#ifdef BUILD_BLAS
    float3 v1 = vertices[indices[param.DispatchThreadID.x * 3]].position.xyz;
    float3 v2 = vertices[indices[param.DispatchThreadID.x * 3 + 1]].position.xyz;
    float3 v3 = vertices[indices[param.DispatchThreadID.x * 3 + 2]].position.xyz;
    
    float3 unit_coord = ((v1 + v2 + v3) / 3.f - push_constants.aabb_min) / max(push_constants.aabb_max - push_constants.aabb_min, 0.00001);
    uint morton_code = MortonCodeFromUnitCoord(unit_coord);
    morton_codes_buffer[param.DispatchThreadID.x] = morton_code;
    indices_buffer[param.DispatchThreadID.x] = param.DispatchThreadID.x;
#endif
}