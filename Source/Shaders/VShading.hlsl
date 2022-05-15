#include "ShaderInterop.hpp"
#include "Common.hlsli"
#include "Math.hlsli"

Texture2D<uint> vbuffer : register(t0);
RWTexture2D<float4> shading : register(u1);
RWTexture2D<float2> normal : register(u2);
ConstantBuffer<Camera> camera : register(b3);
ConstantBuffer<Instance> instances[] : register(b4);
StructuredBuffer<Meshlet> meshlets[] : register(t5);
StructuredBuffer<Vertex> vertices[] : register(t6);
StructuredBuffer<uint> meshlet_vertices[] : register(t7);
StructuredBuffer<uint> meshlet_primitives[] : register(t8);
ConstantBuffer<Material> materials[] : register(b9);

Texture2D<float4> texture_array[] : register(t12);
SamplerState texture_sampler : register(s13);

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
    vbuffer.GetDimensions(extent.x, extent.y);
    
    bary_ddx = (G_dx * H - G * H_dx) * (rcpH * rcpH) * (2.0 / float(extent.x));
    bary_ddy = (G_dy * H - G * H_dy) * (rcpH * rcpH) * (-2.0 / float(extent.y));
}

ShadingState GetShadingState(uint vbuffer_data, float2 uv)
{
    ShadingState sstate;
    
    uint instance_id;
    uint primitive_id;
    uint meshlet_id;
    
    UnPackVBuffer(vbuffer_data, instance_id, meshlet_id, primitive_id);
        
    sstate.matID = instances[instance_id].material;
                
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
    sstate.bary = barycentrics;
    sstate.depth = clip_pos0.z * barycentrics.x + clip_pos1.z * barycentrics.y + clip_pos2.z * barycentrics.z;
    sstate.uv = v0.texcoord.xy * barycentrics.x + v1.texcoord.xy * barycentrics.y + v2.texcoord.xy * barycentrics.z;
    sstate.position = v0.position.xyz * barycentrics.x + v1.position.xyz * barycentrics.y + v2.position.xyz * barycentrics.z;
    sstate.position = mul(transform, float4(sstate.position, 1.0)).xyz;
    sstate.normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
    sstate.normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
    sstate.normal = normalize(mul((float3x3) transform, sstate.normal));
    sstate.tangent = v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z;
    sstate.tangent = normalize(mul((float3x3) transform, sstate.tangent.xyz));
    sstate.dx = v0.texcoord.xy * ddx_barycentrics.x + v1.texcoord.xy * ddx_barycentrics.y + v2.texcoord.xy * ddx_barycentrics.z;
    sstate.dy = v0.texcoord.xy * ddy_barycentrics.x + v1.texcoord.xy * ddy_barycentrics.y + v2.texcoord.xy * ddy_barycentrics.z;
    
    return sstate;
}

void GetMaterial(inout ShadingState sstate)
{
    // Get Metallic Roughness
    if (materials[sstate.matID].type == MetalRoughnessWorkflow)
    {
        float dielectric_specular = (materials[sstate.matID].ior - 1.0) / (materials[sstate.matID].ior + 1.0);
        dielectric_specular *= dielectric_specular;
        
        float perceptual_roughness = 0.0;
        float metallic = 0.0;
        float4 base_color = float4(0.0, 0.0, 0.0, 1.0);
        float3 f0 = float3(dielectric_specular, dielectric_specular, dielectric_specular);
        
        perceptual_roughness = materials[sstate.matID].pbr_roughness_factor;
        metallic = materials[sstate.matID].pbr_metallic_factor;
        if (materials[sstate.matID].pbr_metallic_roughness_texture < MAX_TEXTURE_ARRAY_SIZE)
        {
            float4 mr_sample = texture_array[NonUniformResourceIndex(materials[sstate.matID].pbr_metallic_roughness_texture)].SampleGrad(texture_sampler, sstate.uv, sstate.dx, sstate.dy).rgba;
            perceptual_roughness *= mr_sample.g;
            metallic *= mr_sample.b;
        }
        
        base_color = materials[sstate.matID].pbr_base_color_factor;
        if (materials[sstate.matID].pbr_base_color_texture < MAX_TEXTURE_ARRAY_SIZE)
        {
            base_color *= SRGBtoLINEAR(texture_array[NonUniformResourceIndex(materials[sstate.matID].pbr_base_color_texture)].SampleGrad(texture_sampler, sstate.uv, sstate.dx, sstate.dy).rgba);
        }

        f0 = lerp(float3(dielectric_specular, dielectric_specular, dielectric_specular), base_color.xyz, metallic);
        
        sstate.mat.albedo = base_color.xyz;
        sstate.mat.metallic = metallic;
        sstate.mat.roughness = perceptual_roughness;
        sstate.mat.f0 = f0;
        sstate.mat.alpha = base_color.a;
    }
    
    // Get Specular Glossiness, convert Specular-Glossiness to Metallic-Roughness
    if (materials[sstate.matID].type == SpecularGlossinessWorkflow)
    {
        float perceptual_roughness = 0.0;
        float metallic = 0.0;
        float4 base_color = float4(0.0, 0.0, 0.0, 1.0);
        
        float3 f0 = materials[sstate.matID].pbr_specular_factor;
        perceptual_roughness = 1.0 - materials[sstate.matID].pbr_glossiness_factor;
        
        if (materials[sstate.matID].pbr_specular_glossiness_texture < MAX_TEXTURE_ARRAY_SIZE)
        {
            float4 sg_sample = SRGBtoLINEAR(texture_array[NonUniformResourceIndex(materials[sstate.matID].pbr_specular_glossiness_texture)].SampleGrad(texture_sampler, sstate.uv, sstate.dx, sstate.dy).rgba);
            perceptual_roughness = 1.0 - materials[sstate.matID].pbr_glossiness_factor * sg_sample.a;
            f0 *= sg_sample.rgb;
        }
        
        float3 specular_color = f0;
        float3 one_minus_specular_strength = 1.0 - max(max(f0.r, f0.g), f0.b);
        
        float4 diffuse_color = materials[sstate.matID].pbr_diffuse_factor;
        if (materials[sstate.matID].pbr_diffuse_texture < MAX_TEXTURE_ARRAY_SIZE)
        {
            diffuse_color *= SRGBtoLINEAR(texture_array[NonUniformResourceIndex(materials[sstate.matID].pbr_diffuse_texture)].SampleGrad(texture_sampler, sstate.uv, sstate.dx, sstate.dy).rgba);
        }
        
        base_color.rgb = diffuse_color.rgb * one_minus_specular_strength;

        // Solve metallic
        float specular_brightness = sqrt(0.299 * specular_color.r * specular_color.r + 0.587 * specular_color.g * specular_color.g + 0.114 * specular_color.b * specular_color.b);
        const float min_reflectance = 0.04;
        if (specular_brightness < min_reflectance)
        {
            metallic = 0.0;
        }
        else
        {
            float diffuse_brightness = sqrt(0.299 * diffuse_color.r * diffuse_color.r + 0.587 * diffuse_color.g * diffuse_color.g + 0.114 * diffuse_color.b * diffuse_color.b);
        
            float a = min_reflectance;
            float b = diffuse_brightness * one_minus_specular_strength / (1.0 - min_reflectance) + specular_brightness - 2.0 * min_reflectance;
            float c = min_reflectance - specular_brightness;
            float D = max(b * b - 4.0 * a * c, 0.0);
            metallic = clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
        }
        
        sstate.mat.albedo = base_color.rgb;
        sstate.mat.metallic = metallic;
        sstate.mat.roughness = perceptual_roughness;
        sstate.mat.f0 = f0;
        sstate.mat.alpha = base_color.a;        
    }
    
    // Clamping roughness
    sstate.mat.roughness = max(sstate.mat.roughness, 0.001);
    
    // Emissive
    sstate.mat.emissive = materials[sstate.matID].emissive_factor * materials[sstate.matID].emissive_strength;
    if (materials[sstate.matID].emissive_texture < MAX_TEXTURE_ARRAY_SIZE)
    {
        sstate.mat.emissive *= SRGBtoLINEAR(texture_array[NonUniformResourceIndex(materials[sstate.matID].emissive_texture)].SampleGrad(texture_sampler, sstate.uv, sstate.dx, sstate.dy)).rgb;
    }
    

        
    // Normal
    if (materials[sstate.matID].normal_texture < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 T = sstate.tangent;
        float3 B;
        if (!any(T))
        {
            CreateCoordinateSystem(sstate.normal, T, B);
        }
        else
        {
            float3 B = normalize(cross(sstate.normal, T));
        }
        
        float3x3 TBN = float3x3(T, B, sstate.normal);
        float3 normalVector = texture_array[NonUniformResourceIndex(materials[sstate.matID].normal_texture)].SampleGrad(texture_sampler, sstate.uv, sstate.dx, sstate.dy).rgb;
        normalVector = normalVector * 2.0 - 1.0;
        normalVector = normalize(normalVector);
        sstate.normal = normalize(mul(normalVector, TBN));
    }
}

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    uint2 extent;
    shading.GetDimensions(extent.x, extent.y);

    if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
    {
        return;
    }
    
    float2 screen_texcoord = (float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(extent);
    uint vdata = vbuffer[param.DispatchThreadID.xy];
    
    if (vdata == 0xffffffff)
    {
        shading[param.DispatchThreadID.xy] = float4(0.0, 0.0, 0.0, 1.0);
        normal[param.DispatchThreadID.xy] = PackNormal(float3(0.0, 0.0, 0.0));
        return;
    }
    
    ShadingState sstate = GetShadingState(vdata, screen_texcoord);
    GetMaterial(sstate);
    
    shading[param.DispatchThreadID.xy] = float4(sstate.mat.albedo, 1.0);
    normal[param.DispatchThreadID.xy] = PackNormal(sstate.normal.rgb);
}