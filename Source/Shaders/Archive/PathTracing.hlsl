#include "Random.hlsli"
#include "Common.hlsli"
#include "Math.hlsli"
#include "ShaderInterop.hpp"

RWTexture2D<float4> shading : register(u0);
//RaytracingAccelerationStructure topLevelAS : register(t1);
ConstantBuffer<Camera> camera : register(b2);
Texture2D<uint> vbuffer : register(t3);
ConstantBuffer<Instance> instances[] : register(b4);
StructuredBuffer<Meshlet> meshlets[] : register(t5);
StructuredBuffer<Vertex> vertices[] : register(t6);
StructuredBuffer<uint> meshlet_vertices[] : register(t7);
StructuredBuffer<uint> meshlet_primitives[] : register(t8);
ConstantBuffer<Material> materials[] : register(b9);

Texture2D<float4> texture_array[] : register(t12);
SamplerState texture_sampler : register(s13);

struct VertexAttibute
{
    float3 position;
    float2 uv;
    float3 normal;
    float3 tangent;
    float depth;
    
    float2 dx;
    float2 dy;
    float3 bary;
    
    uint matID;
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
    vbuffer.GetDimensions(extent.x, extent.y);
    
    bary_ddx = (G_dx * H - G * H_dx) * (rcpH * rcpH) * (2.0 / float(extent.x));
    bary_ddy = (G_dy * H - G * H_dy) * (rcpH * rcpH) * (-2.0 / float(extent.y));
}

VertexAttibute GetVertexAttribute(uint instance_id, uint meshlet_id, uint primitive_id, float2 uv)
{
    VertexAttibute attribute;
        
    attribute.matID = instances[instance_id].material;
                
    uint mesh_id = instances[instance_id].mesh;
        
    float4x4 transform = instances[instance_id].transform;
    Meshlet meshlet = meshlets[mesh_id][meshlet_id];
    
    uint i0, i1, i2;
    UnPackTriangle(meshlet_primitives[mesh_id][primitive_id + meshlet.primitive_offset], i0, i1, i2);
    i0 = meshlet_vertices[mesh_id][meshlet.vertex_offset + i0];
    i1 = meshlet_vertices[mesh_id][meshlet.vertex_offset + i1];
    i2 = meshlet_vertices[mesh_id][meshlet.vertex_offset + i2];
    
    Vertex v0 = vertices[mesh_id][i0];
    Vertex v1 = vertices[mesh_id][i1];
    Vertex v2 = vertices[mesh_id][i2];
    
    float4 clip_pos0 = mul(camera.view_projection, mul(transform, float4(v0.position.xyz, 1.0)));
    float4 clip_pos1 = mul(camera.view_projection, mul(transform, float4(v1.position.xyz, 1.0)));
    float4 clip_pos2 = mul(camera.view_projection, mul(transform, float4(v2.position.xyz, 1.0)));

    float2 pixel_clip = uv * 2.0 - 1.0;
    pixel_clip.y *= -1;
    
    float3 barycentrics;
    float3 ddx_barycentrics;
    float3 ddy_barycentrics;
    
    ComputeBarycentrics(pixel_clip, clip_pos0, clip_pos1, clip_pos2, barycentrics, ddx_barycentrics, ddy_barycentrics);
    attribute.bary = barycentrics;
    attribute.depth = clip_pos0.z * barycentrics.x + clip_pos1.z * barycentrics.y + clip_pos2.z * barycentrics.z;
    attribute.uv = v0.texcoord.xy * barycentrics.x + v1.texcoord.xy * barycentrics.y + v2.texcoord.xy * barycentrics.z;
    attribute.position = v0.position.xyz * barycentrics.x + v1.position.xyz * barycentrics.y + v2.position.xyz * barycentrics.z;
    attribute.position = mul(transform, float4(attribute.position, 1.0)).xyz;
    attribute.normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
    attribute.normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
    attribute.normal = normalize(mul((float3x3) transform, attribute.normal));
    attribute.tangent = v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z;
    attribute.tangent = normalize(mul((float3x3) transform, attribute.tangent.xyz));
    attribute.dx = v0.texcoord.xy * ddx_barycentrics.x + v1.texcoord.xy * ddx_barycentrics.y + v2.texcoord.xy * ddx_barycentrics.z;
    attribute.dy = v0.texcoord.xy * ddy_barycentrics.x + v1.texcoord.xy * ddy_barycentrics.y + v2.texcoord.xy * ddy_barycentrics.z;
    
    return attribute;
}

MaterialInfo GetMaterial(inout VertexAttibute attribute)
{
    MaterialInfo mat;
        
    mat.metallic = materials[attribute.matID].pbr_metallic_factor;
    mat.roughness = materials[attribute.matID].pbr_roughness_factor;
    mat.emissive = materials[attribute.matID].emissive_factor * materials[attribute.matID].emissive_strength;
    
    if (materials[attribute.matID].type == MetalRoughnessWorkflow)
    {
        if (materials[attribute.matID].pbr_base_color_texture < MAX_TEXTURE_ARRAY_SIZE)
        {
            float4 base_color = texture_array[NonUniformResourceIndex(materials[attribute.matID].pbr_base_color_texture)].SampleGrad(texture_sampler, attribute.uv, attribute.dx, attribute.dy).rgba;
            mat.albedo = pow(base_color.xyz, float3(2.2, 2.2, 2.2)) * materials[attribute.matID].pbr_base_color_factor.rgb;
        }
    }

    if (materials[attribute.matID].emissive_texture < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 emissive = texture_array[NonUniformResourceIndex(materials[attribute.matID].emissive_texture)].SampleGrad(texture_sampler, attribute.uv, attribute.dx, attribute.dy).rgb;
        emissive = pow(emissive, float3(2.2, 2.2, 2.2));
        mat.emissive *= emissive;
    }
        
    if (materials[attribute.matID].normal_texture < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 T = attribute.tangent;
        float3 B;
        if (!any(T))
        {
            CreateCoordinateSystem(attribute.normal, T, B);
        }
        else
        {
            float3 B = normalize(cross(attribute.normal, T));
        }
        
        float3x3 TBN = float3x3(T, B, attribute.normal);
        float3 normalVector = texture_array[NonUniformResourceIndex(materials[attribute.matID].normal_texture)].SampleGrad(texture_sampler, attribute.uv, attribute.dx, attribute.dy).rgb;
        normalVector = normalVector * 2.0 - 1.0;
        normalVector = normalize(normalVector);
        attribute.normal = normalize(mul(normalVector, TBN));
    }
    
    return mat;
}

[shader("raygeneration")]
void RayGen()
{
    uint2 launch_index = DispatchRaysIndex().xy;
    uint2 launch_dimensions = DispatchRaysDimensions().xy;
    const float2 pixel_center = float2(launch_index.xy) + float2(0.5, 0.5);
    
    PCGSampler random_sampler;
    random_sampler.Init(launch_dimensions, launch_index, camera.frame_count);

    float2 jitter = (random_sampler.Get2D() - float2(0.5, 0.5));
    float2 screen_coords = (pixel_center + jitter) / float2(launch_dimensions.xy);
    
    // Get first bounce result from visibility buffer
    uint vdata = vbuffer[launch_index.xy];
    
    if (vdata == 0xffffffff)
    {
        shading[launch_index] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    uint instance_id;
    uint primitive_id;
    uint meshlet_id;
    UnPackVBuffer(vdata, instance_id, meshlet_id, primitive_id);
    
    VertexAttibute attribute = GetVertexAttribute(instance_id, meshlet_id, primitive_id, screen_coords);
    MaterialInfo material_info = GetMaterial(attribute);
    
    // TODO: Do ray tracing
    
    
    float3 radiance = material_info.albedo;
    // Temporal Accumulation
    if (camera.frame_count == 0)
    {
        shading[launch_index] = float4(radiance, 1.0);
    }
    else
    {
        float3 prev_color = shading[launch_index].rgb;
        float3 accumulated_color = float3(0.0, 0.0, 0.0);
        if ((isnan(prev_color.x) || isnan(prev_color.y) || isnan(prev_color.z)))
        {
            accumulated_color = radiance;
        }
        else
        {
            accumulated_color = lerp(prev_color, radiance, 1.0 / float(camera.frame_count + 1));
        }

        shading[launch_index] = float4(accumulated_color, 1.0);
    }
    
}