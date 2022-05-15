#ifndef __BXDF_HLSL
#define __BXDF_HLSL

#include "Common.hlsli"
#include "Random.hlsli"
#include "Constants.hlsli"

struct BSDFSampleDesc
{
    float3 L;
    float3 f;
    float pdf;
};

struct LightSampleDesc
{
    float3 surface_pos;
    float3 normal;
    float3 emission;
    float pdf;
};

struct Clearcoat
{
    
};

struct Flakes
{
    
};

struct Sheen
{
    
};

struct Emission
{
    
};








float3 ImportanceSampleGTR1(float rgh, float r1, float r2)
{
    float a = max(0.001, rgh);
    float a2 = a * a;
        
    float phi = r1 * PI * 2.0;
        
    float cos_theta = sqrt((1.0 - pow(a2, 1.0 - r1)) / (1.0 - a2));
    float sin_theta = clamp(sqrt(1.0 - (cos_theta * cos_theta)), 0.0, 1.0);
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);

    return float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}
    
float3 ImportanceSampleGTR2(float rgh, float r1, float r2)
{
    float a = max(0.001, rgh);
        
    float phi = r1 * PI * 2.0;
        
    float cos_theta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    float sin_theta = clamp(sqrt(1.0 - (cos_theta * cos_theta)), 0.0, 1.0);
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
        
    return float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}
    
float3 ImportanceSampleGTR2Aniso(float ax, float ay, float r1, float r2)
{
    float phi = r1 * PI * 2;
    float sin_phi = ay * sin(phi);
    float cos_phi = ax * cos(phi);
    float tan_theta = sqrt(r2 / (1.0 - r2));
    return float3(tan_theta * cos_phi, tan_theta * sin_phi, 1.0);
}
    
float SchlickFresnel(float u)
{
    float m = clamp(1.0 - u, 0.0, 1.0);
    float m2 = m * m;
    return m2 * m2 * m;
}
    
float DielectricFresnel(float cos_theta_i, float eta)
{
    float sin_theta_t_sq = eta * eta * (1.0 - cos_theta_i * cos_theta_i);
        
        // Total internal reflection
    if (sin_theta_t_sq > 1.0)
    {
        return 1.0;
    }
        
    float cos_theta_t = sqrt(max(1.0 - sin_theta_t_sq, 0.0));
        
    float rs = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);
    float rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

    return 0.5 * (rs * rs + rp * rp);
}
    
float GTR1(float NoH, float a)
{
    if (a >= 1.0)
    {
        return InvPI;
    }
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NoH * NoH;
    return (a2 - 1.0) / (PI * log(a2) * t);
}
    
float GTR2(float NoH, float a)
{
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NoH * NoH;
    return a2 / (PI * t * t);
}
    
float GTR2Aniso(float NoH, float HoX, float HoY, float ax, float ay)
{
    float a = HoX / ax;
    float b = HoY / ay;
    float c = a * a + b * b + NoH * NoH;
    return 1.0 / (PI * ax * ay * c * c);
}
    
float SmithG_GGX(float NoV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NoV * NoV;
    return 1.0 / (NoV * sqrt(a + b - a * b));
}
    
float SmithG_GGXAniso(float NoV, float VoX, float VoY, float ax, float ay)
{
    float a = VoX * ax;
    float b = VoY * ay;
    float c = NoV;
    return 1.0 / (NoV + sqrt(a * a + b * b + c * c));
}
    
float3 CosineSampleHemisphere(float r1, float r2)
{
    float3 dir;
    float r = sqrt(r1);
    float phi = 2.0 * PI * r2;
    dir.x = r * cos(phi);
    dir.y = r * sin(phi);
    dir.z = sqrt(max(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));
    return dir;
}

float3 UniformSampleHemisphere(float r1, float r2)
{
    float r = sqrt(max(0.0, 1.0 - r1 * r1));
    float phi = 2.0 * PI * r2;
    return float3(r * cos(phi), r * sin(phi), r1);
}

float3 UniformSampleSphere(float r1, float r2)
{
    float z = 1.0 - 2.0 * r1;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * r2;
    return float3(r * cos(phi), r * sin(phi), z);
}

float PowerHeuristic(float a, float b)
{
    float t = a * a;
    return t / (b * b + t);
}

float3 SampleEmitter(RayDesc ray, ShadingState sstate, LightSampleDesc light, BSDFSampleDesc bsdf)
{
    float3 Le;
    if (sstate.trace_depth == 0)
    {
        Le = light.emission;
    }
    else
    {
        Le = PowerHeuristic(bsdf.pdf, light.pdf) * light.emission;
    }
    
    return Le;
}

float3 EvalDielectricReflection(ShadingState sstate, float3 V, float3 N, float3 L, float3 H, inout float pdf)
{
    if (dot(N, L) < 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    float F = DielectricFresnel(dot(V, H), sstate.eta);
    float D = GTR2(dot(N, H), sstate.mat.roughness);
    
    pdf = D * dot(N, H) * F / (4.0 * dot(V, H));
    
    float G = SmithG_GGX(abs(dot(N, L)), sstate.mat.roughness) * SmithG_GGX(dot(N, V), sstate.mat.roughness);
    
    return sstate.mat.albedo * F * D * G;
}

float3 EvalSpecular(ShadingState sstate, float3 Cspec0, float3 V, float3 N, float3 L, float3 H, inout float pdf)
{
    if (dot(N, L) < 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    float D = GTR2Aniso(dot(N, H), dot(H, sstate.tangent), dot(H, sstate.bitangent), sstate.mat.ax, sstate.mat.ay);
    
    pdf = D * dot(N, H) / (4.0 * dot(V, H));
    
    float FH = SchlickFresnel(dot(L, H));
    float3 F = lerp(Cspec0, float3(1.0, 1.0, 1.0), FH);
    float G = SmithG_GGXAniso(dot(N, L), dot(L, sstate.tangent), dot(L, sstate.bitangent), sstate.mat.ax, sstate.mat.ay);
    G *= SmithG_GGXAniso(dot(N, V), dot(L, sstate.tangent), dot(L, sstate.bitangent), sstate.mat.ax, sstate.mat.ay);
    return F * D * G;
}

float3 EvalClearCoat(ShadingState sstate, float3 V, float3 N, float3 L, float3 H, inout float pdf)
{
    if (dot(N, L) < 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    float D = GTR1(dot(N, H), sstate.mat.clearcoat_roughness);
    pdf = D * dot(N, H) / (4.0 * dot(V, H));
    
    float FH = SchlickFresnel(dot(L, H));
    float F = lerp(0.04, 10, FH);
    float G = SmithG_GGX(dot(N, L), 0.25) * SmithG_GGX(dot(N, V), 0.25);
    return float3(0.25 * sstate.mat.clearcoat * F * D * G);
}

float3 EvalDiffuse(ShadingState sstate, float3 Csheen, float3 V, float3 N, float3 L, float3 H, inout float pdf)
{
    if (dot(N, L) < 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    pdf = dot(N, L) * (1.0 / PI);
    
    float FL = SchlickFresnel(dot(N, L));
    float FV = SchlickFresnel(dot(N, V));
    float FH = SchlickFresnel(dot(N, H));
    float Fd90 = 0.5 + 2.0 * dot(L, H) * dot(L, H) * sstate.mat.roughness;
    float Fd = lerp(1.0, Fd90, FL) * lerp(1.0, Fd90, FV);
    float3 Fsheen = FH * sstate.mat.sheen * Csheen;
    return ((1.0 / PI) * Fd * (1.0 - sstate.mat.subsurface) * sstate.mat.albedo + Fsheen) * (1.0 - sstate.mat.metallic);
}

float3 EvalSubsurface(ShadingState sstate, float3 V, float3 N, float3 L, inout float pdf)
{
    pdf = 1.0 / (PI * 2.0);
    
    float FL = SchlickFresnel(abs(dot(N, L)));
    float FV = SchlickFresnel(dot(N, V));
    float Fd = (1.0 - 5.0 * FL) * (1.0 - 5.0 * FV);
    return sqrt(sstate.mat.albedo) * sstate.mat.subsurface * (1.0 / PI) * Fd * (1.0 - sstate.mat.metallic) * (1.0 - sstate.mat.transmission);
}

float3 EvalDisney(inout ShadingState sstate, float3 V, float3 N, float3 L, inout float pdf, inout PCGSampler pcg)
{
    sstate.is_subsurface = false;
    pdf = 0.0;
    float3 f = float3(0.0, 0.0, 0.0);
    
    float r1 = pcg.Get1D();
    float r2 = pcg.Get2D();
    
    float diffuse_ratio = 0.5 * (1.0 - sstate.mat.metallic);
    float trans_weight = (1.0 - sstate.mat.metallic) * sstate.mat.transmission;
    
    float3 Cdlin = sstate.mat.albedo;
    float Cdlum = dot(float3(0.3, 0.6, 0.1), Cdlin);
    
    float3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : float3(1.0, 1.0, 1.0);
    float3 Cspec0 = lerp(sstate.mat.specular * 0.08 * lerp(float3(1.0, 1.0, 1.0), Ctint, sstate.mat.specular_tint), Cdlin, sstate.mat.metallic);
    float3 Csheen = sstate.mat.sheen_tint;
    
    // BSDF
    if (pcg.Get1D() < trans_weight)
    {
        float3 H = ImportanceSampleGTR2(sstate.mat.roughness, r1, r2);
        H = sstate.tangent * H.x + sstate.bitangent * H.y + N * H.z;
        
        float3 R = reflect(-V, H);
        float F = DielectricFresnel(abs(dot(R, H)), sstate.eta);
        
        if (sstate.mat.thin)
        {
            if (dot(sstate.ffnormal, sstate.normal) < 0.0)
            {
                F = 0;
            }
            sstate.eta = 1.001;
        }
        
        // Reflection
        if (pcg.Get1D() < F)
        {
            L = normalize(R);
            f = EvalDielectricReflection(sstate, V, N, L, H, pdf);
        }
        else // Transmission
        {
            L = normalize(refract(-V, H, sstate.eta));
            f = EvalDielectricReflection(sstate, V, N, L, H, pdf);
        }
        
        f *= trans_weight;
        pdf *= trans_weight;
    }
    else //BRDF
    {
        if (pcg.Get1D() < diffuse_ratio)
        {
            // Diffuse Transmission          
            if (pcg.Get1D() < sstate.mat.subsurface)
            {
                L = UniformSampleHemisphere(r1, r2);
                L = sstate.tangent * L.x + sstate.bitangent * L.y - N * L.z;

                f = EvalSubsurface(sstate, V, N, L, pdf);
                pdf *= sstate.mat.subsurface * diffuse_ratio;
                
                sstate.is_subsurface = true;
            }
            else // Diffuse
            {
                L = CosineSampleHemisphere(r1, r2);
                L = sstate.tangent * L.x + sstate.bitangent * L.y + N * L.z;
                
                float3 H = normalize(L + V);
                
                f = EvalDiffuse(sstate, Csheen, V, N, L, H, pdf);
                pdf *= (1.0 - sstate.mat.subsurface) * diffuse_ratio;
            }
        }
        else //Specular
        {
            float spec_ratio = 1.0 / (1.0 + sstate.mat.clearcoat);
            
            if (pcg.Get1D() < spec_ratio)
            {
                float3 H = ImportanceSampleGTR2Aniso(sstate.mat.ax, sstate.mat.ay, r1, r2);
                H = sstate.tangent * H.x + sstate.bitangent * H.y + N * H.z;
                L = normalize(reflect(-V, H));
                
                f = EvalClearCoat(sstate, V, N, L, H, pdf);
                pdf *= (1.0 - spec_ratio) * (1.0 - diffuse_ratio);
            }
        }
        
        f *= 1.0 - trans_weight;
        pdf *= 1.0 - trans_weight;
    }
    return f;
}

float3 EvalDisney(ShadingState sstate, float3 V, float3 N, float3 L, inout float pdf)
{
    float3 H;
    
    if (dot(N, L) < 0.0)
    {
        H = normalize(L * (1.0 / sstate.eta) + V);
    }
    else
    {
        H = normalize(L + V);
    }
    
    if (dot(N, H) < 0.0)
    {
        H = -H;
    }
    
    float diffuse_ratio = 0.5 * (1.0 - sstate.mat.metallic);
    float spec_ratio = 1.0 / (1.0 + sstate.mat.clearcoat);
    float trans_weight = (1.0 - sstate.mat.metallic) * sstate.mat.transmission;
    
    float3 brdf = float3(0.0, 0.0, 0.0);
    float3 bsdf = float3(0.0, 0.0, 0.0);
    float brdf_pdf = 0.0;
    float bsdf_pdf = 0.0;
    
    // BSDF
    if (trans_weight > 0.0)
    {
        // Transmission
        if (dot(N, L) < 0.0)
        {
            bsdf = EvalDielectricReflection(sstate, V, N, L, H, bsdf_pdf);
        }
        else // Reflection
        {
            bsdf = EvalDielectricReflection(sstate, V, N, L, H, bsdf_pdf);
        }
    }
    
    float m_pdf;
    
    if (trans_weight < 1.0)
    {
        // Subsurface
        if (dot(N, L) < 0.0)
        {
            if (sstate.mat.subsurface > 0.0)
            {
                brdf = EvalSubsurface(sstate, V, N, L, m_pdf);
                brdf_pdf = m_pdf * sstate.mat.subsurface * diffuse_ratio;
            }
        }
        else // BRDF
        {
            float3 Cdlin = sstate.mat.albedo;
            float Cdlum = dot(float3(0.3, 0.6, 0.1), Cdlin);
            
            float3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : float3(1.0, 1.0, 1.0);
            float3 Cspec0 = lerp(sstate.mat.specular * 0.08 * lerp(float3(1.0, 1.0, 1.0), Ctint, sstate.mat.specular_tint), Cdlin, sstate.mat.metallic);
            float3 Csheen = sstate.mat.sheen_tint;
            
            // Diffuse
            brdf += EvalDiffuse(sstate, Csheen, V, N, L, N, m_pdf);
            brdf_pdf += m_pdf * (1.0 - sstate.mat.subsurface) * diffuse_ratio;
            
            // Specular
            brdf += EvalSpecular(sstate, Cspec0, V, N, L, H, m_pdf);
            brdf_pdf += m_pdf * spec_ratio * (1.0 - diffuse_ratio);
            
            // Clearcoat
            brdf += EvalClearCoat(sstate, V, N, L, H, m_pdf);
            brdf_pdf += m_pdf * (1.0 - spec_ratio) * (1.0 - diffuse_ratio);
        }
    }
    
    pdf = lerp(brdf_pdf, bsdf_pdf, trans_weight);
    return lerp(brdf, bsdf, trans_weight);
}

#endif