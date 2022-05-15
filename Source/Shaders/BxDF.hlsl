#ifndef __BXDF_HLSL
#define __BXDF_HLSL

#include "Common.hlsli"
#include "Random.hlsli"
#include "Constants.hlsli"

// Fresnel
float3 F_Schlick(float3 f0, float3 f90, float VoH)
{
    return f0 + (f90 - f0) * pow(clamp(1.0 - VoH, 0.0, 1.0), 5.0);
}

float F_Schlick(float f0, float f90, float VoH)
{
    return f0 + (f90 - f0) * pow(clamp(1.0 - VoH, 0.0, 1.0), 5.0);
}

float F_Schlick(float f0, float VoH)
{
    float f90 = 1.0;
    return F_Schlick(f0, f90, VoH);
}

float3 F_Schlick(float3 f0, float f90, float VoH)
{
    float x = clamp(1.0 - VoH, 0.0, 1.0);
    float x2 = x * x;
    float x5 = x * x2 * x2;
    return f0 + (f90 - f0) * x5;
}

float3 F_Schlick(float3 f0, float VoH)
{
    float f90 = 1.0;
    return F_Schlick(f0, f90, VoH);
}

float3 Schlick_to_F0(float3 f, float3 f90, float VoH)
{
    float x = clamp(1.0 - VoH, 0.0, 1.0);
    float x2 = x * x;
    float x5 = clamp(x * x2 * x2, 0.0, 0.9999);
    return (f - f90 * x5) / (1.0 - x5);
}

float Schlick_to_F0(float f, float3 f90, float VoH)
{
    float x = clamp(1.0 - VoH, 0.0, 1.0);
    float x2 = x * x;
    float x5 = clamp(x * x2 * x2, 0.0, 0.9999);
    return (f - f90 * x5) / (1.0 - x5);
}

float3 Schlick_to_F0(float3 f, float VoH)
{
    return Schlick_to_F0(f, float3(1.0, 1.0, 1.0), VoH);
}

float Schlick_to_F0(float f, float VoH)
{
    return Schlick_to_F0(f, 1.0, VoH);
}

// Smith Joint GGX
float V_GGX(float NoL, float NoV, float alpha_roughness)
{
    float alpha_roughness2 = alpha_roughness * alpha_roughness;
    
    float GGXV = NoL * sqrt(NoV * NoV * (1.0 - alpha_roughness2) + alpha_roughness2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0 - alpha_roughness2) + alpha_roughness2);
    
    float GGX = GGXV + GGXL;
    if (GGX > 0.0)
    {
        return 0.5 / GGX;
    }
    return 0.0;
}

float D_GGX(float NoH, float alpha_roughness)
{
    float alpha_roughness2 = alpha_roughness * alpha_roughness;
    float f = (NoH * NoH) * (alpha_roughness2 - 1.0) + 1.0;
    return alpha_roughness2 / (PI * f * f);
}

float LambdaSheenNumericHelper(float x, float alphaG)
{
    float one_minus_alpha2 = (1.0 - alphaG) * (1.0 - alphaG);
    float a = lerp(21.5473, 25.3245, one_minus_alpha2);
    float b = lerp(3.82987, 3.32435, one_minus_alpha2);
    float c = lerp(0.19823, 0.16801, one_minus_alpha2);
    float d = lerp(-1.97760, -1.27393, one_minus_alpha2);
    float e = lerp(-4.32054, -4.85967, one_minus_alpha2);
    return a / (1.0 + b * pow(x, c)) + d * x + e;
}

float LambdaSheen(float cos_theta, float alphaG)
{
    if (abs(cos_theta) < 0.5)
    {
        return exp(LambdaSheenNumericHelper(cos_theta, alphaG));
    }
    else
    {
        return exp(2.0 * LambdaSheenNumericHelper(0.5, alphaG) - LambdaSheenNumericHelper(1.0 - cos_theta, alphaG));
    }
}

float V_Sheen(float NoL, float NoV, float sheen_roughness)
{
    sheen_roughness = max(sheen_roughness, 0.000001);
    float alphaG = sheen_roughness * sheen_roughness;
    return clamp(1.0 / ((1.0 + LambdaSheen(NoV, alphaG) + LambdaSheen(NoL, alphaG)) *
        (4.0 * NoV * NoL)), 0.0, 1.0);
}

float D_Charlie(float sheen_roughness, float NoH)
{
    sheen_roughness = max(sheen_roughness, 0.000001); //clamp (0,1]
    float alphaG = sheen_roughness * sheen_roughness;
    float invR = 1.0 / alphaG;
    float cos2h = NoH * NoH;
    float sin2h = 1.0 - cos2h;
    return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * PI);
}

float3 BRDF_Lambertian(float3 f0, float3 f90, float3 diffuse_color, float specular_weight, float VoH)
{
    return (1.0 - specular_weight * F_Schlick(f0, f90, VoH)) * (diffuse_color / PI);
}

float3 BRDF_LambertianIridescence(float3 f0, float3 f90, float3 iridescence_fresnel, float iridescence_factor, float3 diffuse_color, float specular_weight, float VoH)
{
    float iridescence_max = max(max(iridescence_fresnel.r, iridescence_fresnel.g), iridescence_fresnel.b);
    float3 schlick_fresnel = F_Schlick(f0, f90, VoH);
    float3 F = lerp(schlick_fresnel, float3(iridescence_max, iridescence_max, iridescence_max), iridescence_factor);
    return (1.0 - specular_weight * F) * (diffuse_color / PI);
}

float3 BRDF_SpecularGGX(float3 f0, float3 f90, float alpha_roughness, float specular_weight, float VoH, float NoL, float NoV, float NoH)
{
    float3 F = F_Schlick(f0, f90, VoH);
    float Vis = V_GGX(NoL, NoV, alpha_roughness);
    float D = D_GGX(NoH, alpha_roughness);
    return specular_weight * F * Vis * D;
}

float3 BRDF_SpecularGGXIridescence(float3 f0, float3 f90, float3 iridescence_fresnel, float alpha_roughness, float iridescence_factor, float specular_weight, float VoH, float NoL, float NoV, float NoH)
{
    float3 F = lerp(F_Schlick(f0, f90, VoH), iridescence_fresnel, iridescence_factor);
    float Vis = V_GGX(NoL, NoV, alpha_roughness);
    float D = D_GGX(NoH, alpha_roughness);
    return specular_weight * F * Vis * D;
}

float3 BRDF_SpecularSheen(float3 sheen_color, float sheen_roughness, float NoL, float NoV, float NoH)
{
    float sheen_distribution = D_Charlie(sheen_roughness, NoH);
    float sheen_visibility = V_Sheen(NoL, NoV, sheen_roughness);
    return sheen_color * sheen_distribution * sheen_visibility;
}

#endif