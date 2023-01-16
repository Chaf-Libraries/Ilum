#ifndef MICROFACET_HLSLI
#define MICROFACET_HLSLI

#include "../Math.hlsli"

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

    wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * float3(n);
    // Provide relative IOR along ray to caller
    etap = eta;
    
    return true;
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
        // Transform _w_ to hemispherical configuration
        float3 wh = normalize(float3(alpha_x * w.x, alpha_y * w.y, w.z));
        if (wh.z < 0)
        {
            wh = -wh;
        }
        
        // Find orthonormal basis for visible normal sampling
        float3 T1 = (wh.z < 0.99999f) ? normalize(cross(float3(0, 0, 1), wh))
                                        : float3(1, 0, 0);
        float3 T2 = cross(wh, T1);

        // Generate uniformly distributed points on the unit disk
        float2 p = UniformSampleDisk(u);

        // Warp hemispherical projection for visible normal sampling
        float h = sqrt(1 - Sqr(p.x));
        p.y = lerp((1 + wh.z) / 2, h, p.y);

        // Reproject to hemisphere and transform normal to ellipsoid configuration
        float pz = sqrt(max(0, 1 - LengthSquared(float2(p))));
        float3 nh = p.x * T1 + p.y * T2 + pz * wh;
        return normalize(float3(alpha_x * nh.x, alpha_y * nh.y, max(1e-6f, nh.z)));
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