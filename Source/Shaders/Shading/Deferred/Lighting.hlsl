#include "../../Common.hlsli"

Texture2D<uint> VBuffer : register(t0);
ConstantBuffer<Camera> camera : register(b1);
StructuredBuffer<Instance> instances : register(t2);
StructuredBuffer<Meshlet> meshlets : register(t3);
StructuredBuffer<Vertex> vertices : register(t4);
StructuredBuffer<uint> meshlet_vertices : register(t5);
StructuredBuffer<uint> meshlet_indices : register(t6);
RWTexture2D<float4> Lighting : register(u7);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

void ComputeBarycentrics(float2 uv, float4 v0, float4 v1, float4 v2, out float3 bary, out float3 bary_ddx, out float3 bary_ddy)
{
    float3 pos0 = v0.xyz / v0.w;
    float3 pos1 = v1.xyz / v1.w;
    float3 pos2 = v2.xyz / v2.w;
    
    float3 rcp_w = rcp(float3(v0.w, v1.w, v2.w));
    
    float3 pos120X = float3(pos1.x, pos2.x, pos0.x);
    float3 pos120Y = float3(pos1.y, pos2.y, pos0.y);
    float3 pos201X = float3(pos2.x, pos0.x, pos1.x);
    float3 pos201Y = float3(pos2.y, pos0.y, pos1.y);
    
    float3 C_dx = pos201Y - pos120Y;
    float3 C_dy = pos120X - pos201X;
    
    float3 C = C_dx * (uv.x - pos120X) + C_dy * (uv.y - pos120Y);
    float3 G = C * rcp_w;
    
    float H = dot(C, rcp_w);
    float rcpH = rcp(H);
    
    bary = G * rcpH;
    
    float3 G_dx = C_dx * rcp_w;
    float3 G_dy = C_dy * rcp_w;

    float H_dx = dot(C_dx, rcp_w);
    float H_dy = dot(C_dy, rcp_w);
    
    uint2 extent;
    VBuffer.GetDimensions(extent.x, extent.y);
    
    bary_ddx = (G_dx * H - G * H_dx) * (rcpH * rcpH) * (2.0 / float(extent.x));
    bary_ddy = (G_dy * H - G * H_dy) * (rcpH * rcpH) * (-2.0 / float(extent.y));
}

struct VBufferAttribute
{
    float3 position;
    float2 uv;
    float3 normal;
    float3 tangent;
    
    float2 dx;
    float2 dy;
    float3 bary;
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

VBufferAttribute GetVBufferAttribute(uint vbuffer_data, float2 uv)
{
    uint instance_id;
    uint primitive_id;
    uint meshlet_id;
    
    UnPackVBuffer(vbuffer_data, instance_id, meshlet_id, primitive_id);
    
    VBufferAttribute attribute;
    
    Meshlet meshlet = meshlets[meshlet_id];
    Instance instance = instances[instance_id];
    
    uint vertex_idx[3];
    
    for (int j = primitive_id * 3; j < primitive_id * 3 + 3; j++)
    {
        uint a = (meshlet.meshlet_index_offset + j) / 4;
        uint b = (meshlet.meshlet_index_offset + j) % 4;
        uint idx = (meshlet_indices[a] & 0x000000ffU << (8 * b)) >> (8 * b);
        vertex_idx[j % 3] = idx;
    }
    
    vertex_idx[0] = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + vertex_idx[0]];
    vertex_idx[1] = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + vertex_idx[1]];
    vertex_idx[2] = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + vertex_idx[2]];
    
    Vertex v0 = vertices[vertex_idx[0]];
    Vertex v1 = vertices[vertex_idx[1]];
    Vertex v2 = vertices[vertex_idx[2]];
    
    float4 clip_pos0 = mul(camera.view_projection, mul(instance.transform, float4(v0.position.xyz, 1.0)));
    float4 clip_pos1 = mul(camera.view_projection, mul(instance.transform, float4(v1.position.xyz, 1.0)));
    float4 clip_pos2 = mul(camera.view_projection, mul(instance.transform, float4(v2.position.xyz, 1.0)));
    
    uint2 extent;
    VBuffer.GetDimensions(extent.x, extent.y);
    float2 pixel_clip = uv * 2.0 - 1.0;
    
    float3 barycentrics;
    float3 ddx_barycentrics;
    float3 ddy_barycentrics;
    
    ComputeBarycentrics(pixel_clip, clip_pos0, clip_pos1, clip_pos2, barycentrics, ddx_barycentrics, ddy_barycentrics);
    attribute.bary = barycentrics;
    attribute.uv = vertices[0].uv.xy * barycentrics.x + vertices[1].uv.xy * barycentrics.y + vertices[2].uv.xy * barycentrics.z;
    attribute.position = vertices[0].position.xyz * barycentrics.x + vertices[1].position.xyz * barycentrics.y + vertices[2].position.xyz * barycentrics.z;
    attribute.normal = vertices[0].normal.xyz * barycentrics.x + vertices[1].normal.xyz * barycentrics.y + vertices[2].normal.xyz * barycentrics.z;
    attribute.normal = normalize(mul(attribute.normal, (float3x3) instance.transform));
    
    
    uint mhash = hash(instance_id);
    float3 mcolor = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
    
    attribute.normal = mcolor;
    return attribute;
}

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    uint2 extent;
    VBuffer.GetDimensions(extent.x, extent.y);

    if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
    {
        return;
    }

    float2 pixel_size = 1.0 / float2(extent);

    float2 uv = (float2(param.DispatchThreadID.xy + float2(0.5, 0.5))) / float2(extent);
    
    //VBufferAttribute attribute = GetVBufferAttribute(VBuffer[param.DispatchThreadID.xy].r, uv);
    uint instance_id;
    uint primitive_id;
    uint meshlet_id;
    
    UnPackVBuffer(VBuffer[param.DispatchThreadID.xy].r, instance_id, meshlet_id, primitive_id);
    
    VBufferAttribute attribute;
    
    Meshlet meshlet = meshlets[meshlet_id];
    Instance instance = instances[instance_id];
    
    uint vertex_idx[3];
    
    for (int j = primitive_id * 3; j < primitive_id * 3 + 3; j++)
    {
        uint a = (meshlet.meshlet_index_offset + j) / 4;
        uint b = (meshlet.meshlet_index_offset + j) % 4;
        uint idx = (meshlet_indices[a] & 0x000000ffU << (8 * b)) >> (8 * b);
        vertex_idx[j % 3] = idx;
    }
    
    vertex_idx[0] = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + vertex_idx[0]];
    vertex_idx[1] = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + vertex_idx[1]];
    vertex_idx[2] = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + vertex_idx[2]];
    
    Vertex v0 = vertices[vertex_idx[0]];
    Vertex v1 = vertices[vertex_idx[1]];
    Vertex v2 = vertices[vertex_idx[2]];
    
    float4 clip_pos0 = mul(camera.view_projection, mul(instance.transform, float4(v0.position.xyz, 1.0)));
    float4 clip_pos1 = mul(camera.view_projection, mul(instance.transform, float4(v1.position.xyz, 1.0)));
    float4 clip_pos2 = mul(camera.view_projection, mul(instance.transform, float4(v2.position.xyz, 1.0)));
    
    float3 color = float3((clip_pos0 + clip_pos1 + clip_pos2).xy / 3, 0.0);
    
    uint mhash = hash(vertex_idx[0] + vertex_idx[1] + vertex_idx[2]);
    float3 mcolor = float3(float(hash(mhash) & 255), float((hash(mhash) >> 8) & 255), float((hash(mhash) >> 16) & 255)) / 255.0;
    
    Lighting[param.DispatchThreadID.xy] = float4(color, 1.0);

}