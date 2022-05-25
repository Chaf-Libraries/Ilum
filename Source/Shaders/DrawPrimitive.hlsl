//#define DRAW_AABB

//#ifdef DRAW_AABB
[[vk::push_constant]]
struct
{
    float4x4 view_projection;
    float4 aabb_min;
    float4 aabb_max;
    float4 color;
} push_constants;
//#endif

struct VertexIn
{
    uint Vertex_ID : SV_VertexID;
};

struct VertexOut
{
    float4 Position : SV_Position;
    float3 Color : COLOR0;
};

struct FragmentOut
{
    float4 Color : SV_Target0;
};

VertexOut VSmain(VertexIn vert_in)
{
    VertexOut vert_out;

//#ifdef DRAW_AABB
    const float3 aabb_verts[] =
    {
        float3(-1, -1, -1), float3(-1, 1, -1), float3(1, 1, -1), float3(1, -1, -1), float3(-1, -1, -1),
        float3(-1, -1, 1), float3(-1, 1, 1), float3(1, 1, 1), float3(1, -1, 1), float3(-1, -1, 1),
        float3(-1, 1, 1), float3(-1, 1, -1), float3(1, 1, -1), float3(1, 1, 1), float3(1, -1, 1), float3(1, -1, -1)
    };
    
    float3 aabb_vert = aabb_verts[vert_in.Vertex_ID];
    aabb_vert.x = aabb_vert.x > 0 ? push_constants.aabb_max.x : push_constants.aabb_min.x;
    aabb_vert.y = aabb_vert.y > 0 ? push_constants.aabb_max.y : push_constants.aabb_min.y;
    aabb_vert.z = aabb_vert.z > 0 ? push_constants.aabb_max.z : push_constants.aabb_min.z;
    
    vert_out.Position = mul(push_constants.view_projection, float4(aabb_vert, 1.0));
    vert_out.Color = push_constants.color.rgb;
//#endif
    
    return vert_out;
}

FragmentOut PSmain(VertexOut vert)
{
    FragmentOut frag_out;
    frag_out.Color = float4(vert.Color, 1.0);
    return frag_out;
}