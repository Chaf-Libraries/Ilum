#ifndef __LIGHTS_HLSL__
#define __LIGHTS_HLSL__

#include "ShaderInterop.hpp"
#include "ShadingState.hlsli"
#include "BxDF.hlsli"
#include "Math.hlsli"

ConstantBuffer<DirectionalLight> directional_lights[] : register(b0, space3);
ConstantBuffer<SpotLight> spot_lights[] : register(b1, space3);
ConstantBuffer<PointLight> point_lights[] : register(b2, space3);
ConstantBuffer<AreaLight> area_lights[] : register(b3, space3);

Texture2D<float4> spot_shadows[] : register(t4, space3);
Texture2DArray<float4> directional_shadows[] : register(t5, space3);
TextureCube<float4> point_shadows[] : register(t6, space3);

struct VisibilityTester
{
    float3 from;
    float3 direction;
    float distance;
};

float3 Eval_Light(DirectionalLight light, float3 shading_point, out float3 L)
{
    L = normalize(-light.direction);
    return light.color.rgb * light.intensity;
}

float3 Eval_Light(SpotLight light, float3 shading_point, out float3 L)
{
    L = normalize(light.position - shading_point);
    
    float light_angle_scale = 1.0 / max(0.001, cos(light.cut_off) - cos(light.outer_cut_off));
    float light_angle_offset = -cos(light.outer_cut_off) * light_angle_scale;
    float cd = max(dot(light.direction, L), 0.0);
    float angular_attenuation = saturate(cd * light_angle_scale + light_angle_offset);

    return light.color.rgb * light.intensity * angular_attenuation * angular_attenuation;
}

float3 Eval_Light(PointLight light, float3 shading_point, out float3 L)
{
    L = normalize(light.position - shading_point);

    float d = length(light.position - shading_point);
    float attenuation = max(min(1.0 - pow(d / light.range, 4.0), 1.0), 0.0) / (d * d);

    return light.color.rgb * light.intensity * attenuation;
}

float3 Sample_Light(DirectionalLight light, float3 shading_point, float2 pcg, out float3 L, out float pdf, out VisibilityTester vis)
{
    pdf = 1.0;
    
    vis.from = shading_point;
    vis.direction = -light.direction;
    vis.distance = Infinity;
    
    return Eval_Light(light, shading_point, L);
}

float3 Sample_Light(SpotLight light, float3 shading_point, float2 pcg, out float3 L, out float pdf, out VisibilityTester vis)
{
    if (light.radius == 0.0)
    {
        pdf = 1.0;
    
        vis.from = shading_point;
        vis.direction = normalize(light.position - shading_point);
        vis.distance = length(light.position - shading_point);
        
        return Eval_Light(light, shading_point, L);
    }
    else
    {
        float3 p = light.radius * UniformSampleSphere(pcg) + light.position;
        pdf = 1.0 / (2 * PI * light.radius * light.radius);
        
        float3 lighting = Eval_Light(light, shading_point, L);
        L = normalize(p - shading_point);
        
        vis.from = shading_point;
        vis.direction = normalize(p - shading_point);
        vis.distance = length(p - shading_point);
        
        return lighting;
    }
}

float3 Sample_Light(PointLight light, float3 shading_point, float2 pcg, out float3 L, out float pdf, out VisibilityTester vis)
{
    if (light.radius == 0.0)
    {
        pdf = 1.0;
        
        vis.from = shading_point;
        vis.direction = normalize(light.position - shading_point);
        vis.distance = length(light.position - shading_point);
        
        return Eval_Light(light, shading_point, L);
    }
    else
    {
        float3 p = light.radius * UniformSampleSphere(pcg) + light.position;
        pdf = 1.0 / (2 * PI * light.radius * light.radius);
        
        float3 lighting = Eval_Light(light, shading_point, L);
        L = normalize(p - shading_point);
        
        vis.from = shading_point;
        vis.direction = normalize(p - shading_point);
        vis.distance = length(p - shading_point);
        
        return lighting;
    }
}

float3 GetPunctualRadianceClearCoat(float3 N, float3 V, float3 L, float3 H, float VoH, float3 f0, float3 f90, float clearcoat_roughness)
{
    float NoL = clamp(dot(N, L), 0.0, 1.0);
    float NoV = clamp(dot(N, V), 0.0, 1.0);
    float NoH = clamp(dot(N, H), 0.0, 1.0);
    return NoL * Eval_BRDF_SpecularGGX(f0, f90, clearcoat_roughness * clearcoat_roughness, 1.0, VoH, NoL, NoV, NoH);
}

float3 GetPunctualRadianceSheen(float3 sheen_color, float sheen_roughness, float NoL, float NoV, float NoH)
{
    return NoL * Eval_BRDF_SpecularSheen(sheen_color, sheen_roughness, NoL, NoV, NoH);
}

#endif