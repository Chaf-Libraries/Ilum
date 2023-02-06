#ifndef PLASTIC_BSDF_HLSLI
#define PLASTIC_BSDF_HLSLI

#include "BSDF.hlsli"
#include "MicrofacetReflectionBSDF.hlsli"
#include "LambertianReflectionBSDF.hlsli"
#include "../Fresnel.hlsli"
#include "../Scattering.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct PlasticBSDF
{
    LambertianReflectionBSDF lambertian_reflection;
    MicrofacetReflectionBSDF<FresnelDielectric, TrowbridgeReitzDistribution> microfacet_reflection;
    
    void Init(float3 R, float roughness, float anisotropic, float3 normal_)
    {
        FresnelDielectric fresnel;
        TrowbridgeReitzDistribution distribution;
        
        float rough = max(roughness, 0.001);
        float aspect = sqrt(1.0 - anisotropic * 0.9);
        float urough = max(0.001, roughness / aspect);
        float vrough = max(0.001, roughness * aspect);
        
        fresnel.eta_i = 1.f;
        fresnel.eta_t = 1.5f;
        
        distribution.alpha_x = urough;
        distribution.alpha_y = vrough;
        distribution.sample_visible_area = true;
        
        microfacet_reflection.Init(R, distribution, fresnel, normal_);
        lambertian_reflection.Init(R, normal_);
    }
    
    uint Flags()
    {
        return BSDF_GlossyReflection | BSDF_DiffuseReflection;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return lambertian_reflection.Eval(woW, wiW, mode) * 0.5f + microfacet_reflection.Eval(woW, wiW, mode) * 0.5f;
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return lambertian_reflection.PDF(woW, wiW, mode, flags) * 0.5f + microfacet_reflection.PDF(woW, wiW, mode, flags) * 0.5f;
    }
    
    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        if ((uc + u.x + u.y) * 0.33 < 0.5f)
        {
            return lambertian_reflection.Samplef(woW, uc, u, mode, flags);
        }
        else
        {
            return microfacet_reflection.Samplef(woW, uc, u, mode, flags);
        }
    }
};

#endif