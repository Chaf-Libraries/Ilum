#include "../ShaderInterop.hpp"

StructuredBuffer<HierarchyNode> hierarchy_buffer[] : register(t0);
StructuredBuffer<AABB> aabbs_buffer[] : register(t1);

[[vk::push_constant]]
struct
{
    float4x4 view_projection;
    float4x4 transform;
    uint instance_id;
} push_constants;

struct VertexIn
{
    uint vertex_id : SV_VertexID;
};

struct VertexOut
{
    float3 aabb_min : POSITION0;
    float3 aabb_max : POSITION1;
    float3 color : COLOR0;
};

struct GeometryOut
{
    float4 Pos : SV_POSITION;
    float3 Color : COLOR0;
};

struct FragmentOut
{
    float4 Color : SV_Target0;
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

VertexOut VSmain(VertexIn vertex_in)
{
    VertexOut vertex_out;
    AABB aabb = aabbs_buffer[push_constants.instance_id][vertex_in.vertex_id].Transform(push_constants.transform);
    vertex_out.aabb_max = aabb.max_val.xyz;
    vertex_out.aabb_min = aabb.min_val.xyz;
    uint node = vertex_in.vertex_id;
    uint depth = 0;
    while (hierarchy_buffer[push_constants.instance_id][node].parent != ~0U)
    {
        node = hierarchy_buffer[push_constants.instance_id][node].parent;
        depth += 1;
    }
    uint mhash = hash(depth);
    float3 color = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
    
    vertex_out.color = color;
    
    return vertex_out;
}

[maxvertexcount(16)]
void GSmain(point VertexOut verts[1], inout LineStream<GeometryOut> outStream)
{
    const float3 aabb_verts[] =
    {
        float3(-1, -1, -1), float3(-1, 1, -1), float3(1, 1, -1), float3(1, -1, -1), float3(-1, -1, -1),
        float3(-1, -1, 1), float3(-1, 1, 1), float3(1, 1, 1), float3(1, -1, 1), float3(-1, -1, 1),
        float3(-1, 1, 1), float3(-1, 1, -1), float3(1, 1, -1), float3(1, 1, 1), float3(1, -1, 1), float3(1, -1, -1)
    };
    
    for (int i = 0; i < 16; i++)
    {
        float3 aabb_vert = aabb_verts[i];
        aabb_vert.x = aabb_vert.x > 0 ? verts[0].aabb_max.x : verts[0].aabb_min.x;
        aabb_vert.y = aabb_vert.y > 0 ? verts[0].aabb_max.y : verts[0].aabb_min.y;
        aabb_vert.z = aabb_vert.z > 0 ? verts[0].aabb_max.z : verts[0].aabb_min.z;
        
        GeometryOut geo_out;
        geo_out.Pos = mul(push_constants.view_projection, float4(aabb_vert, 1.0));
        geo_out.Color = verts[0].color;
        
        outStream.Append(geo_out);
    }
    outStream.RestartStrip();
}

FragmentOut PSmain(GeometryOut geo)
{
    FragmentOut frag_out;
    frag_out.Color = float4(geo.Color, 1.0);
    return frag_out;
}