#ifndef ATTRIBUTE_HLSLI
#define ATTRIBUTE_HLSLI

struct InstanceAttribute
{
    float instance_id;
    float material_id;
    float primitive_id;
}

struct VertexAttribute
{
    float3 position;
    float3 normal;
    float3 tangent;
    float3 texcoord0;
    float3 texcoord1;
}

static InstanceAttribute instance_attribute;
static VertexAttribute vertex_attribute;

#endif