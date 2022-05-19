#include "ShaderInterop.hpp"
#include "Common.hlsli"
#include "Math.hlsli"
#include "ShadingState.hlsli"

Texture2D<uint> vbuffer : register(t0, space0);
RWTexture2D<float4> shading : register(u1, space0);
RWTexture2D<float2> normal : register(u2, space0);
ConstantBuffer<Camera> camera : register(b3, space0);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

[[vk::push_constant]]
struct
{
    uint point_light_count;
} push_constants;

/*void GetMaterial(inout ShadingState sstate)
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
}*/

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
    
    ShadingState sstate;
    sstate.LoadVisibilityBuffer(vbuffer, param.DispatchThreadID.xy, camera.view_projection);
    
    shading[param.DispatchThreadID.xy] = float4(sstate.normal, 1.0);
    normal[param.DispatchThreadID.xy] = PackNormal(sstate.normal.rgb);
}