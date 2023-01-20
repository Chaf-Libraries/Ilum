#ifndef MICROFACET_HLSLI
#define MICROFACET_HLSLI

#include "../Math.hlsli"
#include "../Interaction.hlsli"

bool Refract(float3 wi, float3 n, float eta, out float etap, out float3 wt)
{
    float cosTheta_i = dot(n, wi);
    // Potentially flip interface orientation for Snell's law
    if (cosTheta_i < 0)
    {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
        n = -n;
    }

    // Compute $\cos\,\theta_\roman{t}$ using Snell's law
    float sin2Theta_i = max(0, 1 - Sqr(cosTheta_i));
    float sin2Theta_t = sin2Theta_i / Sqr(eta);
    // Handle total internal reflection case
    if (sin2Theta_t >= 1)
    {
        return false;
    }
    
    float cosTheta_t = sqrt(1 - sin2Theta_t);

    wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * n;
    // Provide relative IOR along ray to caller
    etap = eta;
    
    return true;
}

float3 Reflect(float3 wo, float3 n)
{
    return -wo + 2 * dot(wo, n) * n;
}

float HenyeyGreenstein(float cosTheta, float g)
{
    float denom = 1 + Sqr(g) + 2 * g * cosTheta;
    return Inv4PI * (1 - Sqr(g)) / (denom * sqrt(denom));
}

float FresnelDielectric(float cosTheta_i, float eta)
{
    cosTheta_i = clamp(cosTheta_i, -1, 1);
    // Potentially flip interface orientation for Fresnel equations
    if (cosTheta_i < 0)
    {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
    }

    // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    float sin2Theta_i = 1 - Sqr(cosTheta_i);
    float sin2Theta_t = sin2Theta_i / Sqr(eta);
    if (sin2Theta_t >= 1)
    {
        return 1.f;
    }
    float cosTheta_t = sqrt(1 - sin2Theta_t);

    float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (Sqr(r_parl) + Sqr(r_perp)) / 2;
}

float FresnelComplex(float cosTheta_i, Complex eta)
{
    Complex cosTheta_i_ = ComplexFromReal(clamp(cosTheta_i, 0, 1));
    // Compute complex $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    Complex sin2Theta_i = Sub(ComplexFromReal(1), Mul(cosTheta_i_, cosTheta_i_));
    Complex sin2Theta_t = Div(sin2Theta_i, Mul(eta, eta));
    Complex cosTheta_t = Sqrt(Sub(ComplexFromReal(1), sin2Theta_t));
    
    Complex r_parl = Div(Sub(Mul(eta, cosTheta_i_), cosTheta_t), Add(Mul(eta, cosTheta_i_), cosTheta_t));
    Complex r_perp = Div(Sub(cosTheta_i_, Mul(eta, cosTheta_t)), Add(Mul(eta, cosTheta_t), cosTheta_i_));

    return (Norm(r_parl) + Norm(r_perp)) / 2;
}

float3 FresnelComplex(float cosTheta_i, float3 eta, float3 k)
{
    float3 result;
    result.x = FresnelComplex(cosTheta_i, CreateComplex(eta.x, k.x));
    result.y = FresnelComplex(cosTheta_i, CreateComplex(eta.y, k.y));
    result.z = FresnelComplex(cosTheta_i, CreateComplex(eta.z, k.z));
    return result;
}

struct TrowbridgeReitzDistribution
{
    float alpha_x, alpha_y;
    
    bool EffectivelySmooth()
    {
        return max(alpha_x, alpha_y) < 1e-3f;
    }
    
    void Init(float ax, float ay)
    {
        alpha_x = ax;
        alpha_y = ay;
        if (!EffectivelySmooth())
        {
            alpha_x = max(alpha_x, 1e-4f);
            alpha_y = max(alpha_y, 1e-4f);
        }
    }
    
    float D(float3 wm)
    {
        float tan2Theta = Tan2Theta(wm);
        if (isinf(tan2Theta))
        {
            return 0;
        }
        float cos4Theta = Sqr(Cos2Theta(wm));
        if (cos4Theta < 1e-16f)
        {
            return 0;
        }
        float e = tan2Theta * (Sqr(CosPhi(wm) / alpha_x) + Sqr(SinPhi(wm) / alpha_y));
        return 1 / (PI * alpha_x * alpha_y * cos4Theta * Sqr(1 + e));
    }
    
    float Lambda(float3 w)
    {
        float tan2Theta = Tan2Theta(w);
        if (isinf(tan2Theta))
        {
            return 0;
        }
        float alpha2 = Sqr(CosPhi(w) * alpha_x) + Sqr(SinPhi(w) * alpha_y);
        return (sqrt(1 + alpha2 * tan2Theta) - 1) / 2;
    }
    
    float G1(float3 w)
    {
        return 1 / (1 + Lambda(w));
    }
    
    float G(float3 wo, float3 wi)
    {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }
    
    float D(float3 w, float3 wm)
    {
        return G1(w) / AbsCosTheta(w) * D(wm) * abs(dot(w, wm));
    }
    
    float PDF(float3 w, float3 wm)
    {
        return D(w, wm);
    }
    
    float3 Sample_wm(float3 w, float2 u)
    {
        // Reference: https://jcgt.org/published/0007/04/01/paper.pdf
        float3 vh = normalize(float3(alpha_x * w.x, alpha_y * w.y, w.z));
        
        float len = vh.x * vh.x + vh.y * vh.y;
        float3 T1 = len > 0 ? float3(-vh.y, -vh.x, 0.f) / sqrt(len) : float3(1, 0, 0);
        float3 T2 = cross(vh, T1);
        
        float2 p = UniformSampleDisk(u);
        float s = 0.5 * (1.0 + vh.z);
        p.y = (1.0 - s) * sqrt(1.0 - p.x * p.x) + s * p.y;
        float3 nh = p.x * T1 + p.y * T2 + sqrt(max(0, 1.0 - p.x * p.x - p.y * p.y)) * vh;
        
        return normalize(float3(alpha_x * nh.x, alpha_y * nh.y, max(0, nh.z)));
    }
    
    float RoughnessToAlpha(float roughness)
    {
        return sqrt(roughness);
    }
    
    void Regularize()
    {
        if (alpha_x < 0.3f)
        {
            alpha_x = clamp(2 * alpha_x, 0.1f, 0.3f);
        }
        if (alpha_y < 0.3f)
        {
            alpha_y = clamp(2 * alpha_y, 0.1f, 0.3f);
        }
    }
};

#endif