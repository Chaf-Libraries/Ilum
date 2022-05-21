#ifndef __LIGHTS_HLSL__
#define __LIGHTS_HLSL__

#include "ShaderInterop.hpp"
#include "ShadingState.hlsli"
#include "BxDF.hlsli"

ConstantBuffer<DirectionalLight> directional_light[] : register(b0, space3);
ConstantBuffer<SpotLight> spot_light[] : register(b1, space3);
ConstantBuffer<PointLight> point_light[] : register(b2, space3);
ConstantBuffer<AreaLight> area_light[] : register(b3, space3);

Texture2D<float4> spot_shadow[] : register(t4, space3);
Texture2DArray<float4> directional_shadow[] : register(t5, space3);
TextureCube<float4> point_shadow[] : register(t6, space3);

float3 Eval_Light(DirectionalLight light, float3 shading_point, out float3 wi)
{
    wi = normalize(-light.direction);
    return light.color.rgb * light.intensity;
}

float3 Eval_Light(SpotLight light, float3 shading_point, out float3 wi)
{
    wi = normalize(light.position - shading_point);
    
    float light_angle_scale = 1.0 / max(0.001, cos(light.cut_off) - cos(light.outer_cut_off));
    float light_angle_offset = -cos(light.outer_cut_off) * light_angle_scale;
    float cd = abs(dot(light.direction, wi));
    float angular_attenuation = saturate(cd * light_angle_scale + light_angle_offset);

    return light.color.rgb * light.intensity * angular_attenuation * angular_attenuation;
}

float3 Eval_Light(PointLight light, float3 shading_point, out float3 wi)
{
    wi = normalize(light.position - shading_point);

    float d = length(light.position - shading_point);
    float attenuation = max(min(1.0 - pow(d / light.range, 4.0), 1.0), 0.0) / (d * d);
                
    return light.color.rgb * light.intensity * attenuation;
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