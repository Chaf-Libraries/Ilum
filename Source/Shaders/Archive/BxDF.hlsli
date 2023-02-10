#ifndef __BXDF_HLSL
#define __BXDF_HLSL

#include "Common.hlsli"
#include "Random.hlsli"
#include "Constants.hlsli"
#include "ShadingState.hlsli"

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

float Schlick_to_F0(float f, float f90, float VoH)
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

float3 Eval_BRDF_Lambertian(float3 f0, float3 f90, float3 diffuse_color, float specular_weight, float VoH)
{
    return (1.0 - specular_weight * F_Schlick(f0, f90, VoH)) * (diffuse_color / PI);
}

float3 Eval_BRDF_LambertianIridescence(float3 f0, float3 f90, float3 iridescence_fresnel, float iridescence_factor, float3 diffuse_color, float specular_weight, float VoH)
{
    float iridescence_max = max(max(iridescence_fresnel.r, iridescence_fresnel.g), iridescence_fresnel.b);
    float3 schlick_fresnel = F_Schlick(f0, f90, VoH);
    float3 F = lerp(schlick_fresnel, float3(iridescence_max, iridescence_max, iridescence_max), iridescence_factor);
    return (1.0 - specular_weight * F) * (diffuse_color / PI);
}

float3 Eval_BRDF_SpecularGGX(float3 f0, float3 f90, float alpha_roughness, float specular_weight, float VoH, float NoL, float NoV, float NoH)
{
    float3 F = F_Schlick(f0, f90, VoH);
    float Vis = V_GGX(NoL, NoV, alpha_roughness);
    float D = D_GGX(NoH, alpha_roughness);
    return specular_weight * F * Vis * D;
}

float3 Eval_BRDF_SpecularGGXIridescence(float3 f0, float3 f90, float3 iridescence_fresnel, float alpha_roughness, float iridescence_factor, float specular_weight, float VoH, float NoL, float NoV, float NoH)
{
    float3 F = lerp(F_Schlick(f0, f90, VoH), iridescence_fresnel, iridescence_factor);
    float Vis = V_GGX(NoL, NoV, alpha_roughness);
    float D = D_GGX(NoH, alpha_roughness);
    return specular_weight * F * Vis * D;
}

float3 Eval_BRDF_SpecularSheen(float3 sheen_color, float sheen_roughness, float NoL, float NoV, float NoH)
{
    float sheen_distribution = D_Charlie(sheen_roughness, NoH);
    float sheen_visibility = V_Sheen(NoL, NoV, sheen_roughness);
    return sheen_color * sheen_distribution * sheen_visibility;
}

// Iridescent Fresnel
static const float3x3 XYZ_TO_REC709 = float3x3(
    3.2404542, -0.9692660, 0.0556434,
    -1.5371385, 1.8760108, -0.2040259,
    -0.4985314, 0.0415560, 1.0572252
);

// F0 -> IOR
float3 Fresnel0ToIOR(float3 fresnel0)
{
    float3 sqrtF0 = sqrt(fresnel0);
    return (1.0 + sqrtF0) / (1.0 - sqrtF0);
}

// IOR -> F0
float3 IORToFresnel0(float3 transmitted_ior, float incident_ior)
{
    return pow((transmitted_ior - incident_ior) / (transmitted_ior + incident_ior), 2.0);
}

float IORToFresnel0(float transmitted_ior, float incident_ior)
{
    return pow((transmitted_ior - incident_ior) / (transmitted_ior + incident_ior), 2.0);
}

// Fresnel equations for dielectric/dielectric interfaces.
// Evaluation XYZ sensitivity curves in Fourier space
float3 Eval_Sensitivity(float OPD, float3 shift)
{
    float phase = 2.0 * PI * OPD * 1.0e-9;
    float3 val = float3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    float3 pos = float3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    float3 var = float3(4.3278e+09, 9.3046e+09, 6.6121e+09);

    float3 xyz = val * sqrt(2.0 * PI * var) * cos(pos * phase + shift) * exp(-phase * phase * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift[0]) * exp(-4.5282e+09 * phase * phase);
    xyz /= 1.0685e-7;

    float3 srgb = mul(XYZ_TO_REC709, xyz);
    return srgb;
}

float3 Iridescent_Fresnel(float outside_ior, float eta2, float cosTheta1, float iridescence_thickness, float3 baseF0)
{
    float3 F_iridescence = 0.0;
    
    // Force iridescenceIor -> outside_ior when thinFilmThickness -> 0.0
    float iridescence_ior = lerp(outside_ior, eta2, smoothstep(0.0, 0.03, iridescence_thickness));
    // Evaluate the cosTheta on the base layer (Senll law)
    float sinTheta2Sq = pow(outside_ior / iridescence_ior, 2.0) * (1.0 - pow(cosTheta1, 2.0));
    float cosTheta2Sq = 1.0 - sinTheta2Sq;
    
    // Handle total internal reflection
    if (cosTheta2Sq < 0.0)
    {
        return 1.0;
    }
    
    float cosTheta2 = sqrt(cosTheta2Sq);
    
    // First Interface
    // Belcour/Barla models
    float R0 = IORToFresnel0(iridescence_ior, outside_ior);
    float R12 = F_Schlick(R0, cosTheta1);
    float R21 = R12;
    float T121 = 1.0 - R12;
    float phi12 = 0.0;
    if (iridescence_ior < outside_ior)
    {
        phi12 = PI;
    }
    float phi21 = PI - phi12;
    
    // Second Interface
    float3 base_ior = Fresnel0ToIOR(baseF0 + 0.0001);
    float3 R1 = IORToFresnel0(base_ior, iridescence_ior);
    float3 R23 = F_Schlick(R1, cosTheta2);
    float3 phi23 = 0.0;
    if (base_ior.x < iridescence_ior)
    {
        phi23.x = PI;
    }
    if (base_ior.y < iridescence_ior)
    {
        phi23.y = PI;
    }
    if (base_ior.y < iridescence_ior)
    {
        phi23.y = PI;
    }
    
    // first-order optical path difference
    float OPD = 2.0 * iridescence_ior * iridescence_thickness * cosTheta1;
    
    // Phase Shift
    float3 phi = phi21 + phi23;
    
    // Compound terms
    float3 R123 = clamp(R12 * R23, 1e-5, 0.9999);
    float3 r123 = sqrt(R123);
    float3 Rs = T121 * T121 * R23 / (1.0 - R123);
    
    // Reflectance term for m = 0 (DC term amplitude)
    float3 C0 = R12 + Rs;
    float3 I = C0;
    
    // Reflectance term for m > 0 (pairs of diracs)
    float3 Cm = Rs - T121;
    for (int m = 1; m < 2; m++)
    {
        Cm *= r123;
        float3 Sm = 2.0 * Eval_Sensitivity(float(m) * OPD, float(m) * phi);
        I += Cm * Sm;
    }
    
    F_iridescence = max(I, 0.0);
    
    return F_iridescence;
}

#endif