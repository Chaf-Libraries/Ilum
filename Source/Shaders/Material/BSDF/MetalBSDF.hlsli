#ifndef METAL_BSDF_HLSLI
#define METAL_BSDF_HLSLI

#include "BSDF.hlsli"
#include "MicrofacetReflectionBSDF.hlsli"
#include "LambertianReflectionBSDF.hlsli"
#include "../Fresnel.hlsli"
#include "../Scattering.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct MetalBSDF
{
    MicrofacetReflectionBSDF<FresnelConductor, TrowbridgeReitzDistribution> microfacet_reflection;
    
    void Init(float3 R, float roughness, float3 eta, float3 k, float anisotropic, float3 normal_)
    {
        FresnelConductor fresnel;
        TrowbridgeReitzDistribution distribution;
        
        float rough = max(roughness, 0.001);
        float aspect = sqrt(1.0 - anisotropic * 0.9);
        float urough = max(0.001, roughness / aspect);
        float vrough = max(0.001, roughness * aspect);
        
        fresnel.eta_i = 1.f;
        fresnel.eta_t = eta;
        fresnel.k = k;
        
        distribution.alpha_x = urough;
        distribution.alpha_y = vrough;
        distribution.sample_visible_area = true;

        microfacet_reflection.Init(R, distribution, fresnel, normal_);
    }
    
    uint Flags()
    {
        return BSDF_GlossyReflection;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return microfacet_reflection.Eval(woW, wiW, mode);
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return microfacet_reflection.PDF(woW, wiW, mode, flags);
    }
    
    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        return microfacet_reflection.Samplef(woW, uc, u, mode, flags);
    }
};

#endif