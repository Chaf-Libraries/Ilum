#ifndef SUBSTRATE_BSDF_HLSLI
#define SUBSTRATE_BSDF_HLSLI

#include "BSDF.hlsli"
#include "FresnelBlendBSDF.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct SubstrateBSDF
{
    FresnelBlendBSDF<TrowbridgeReitzDistribution> fresnel_blend;
    
    void Init(float3 Rd, float3 Rs, float roughness, float anisotropic, float3 normal_)
    {
        TrowbridgeReitzDistribution distribution;
        
        float rough = max(roughness, 0.001);
        float aspect = sqrt(1.0 - anisotropic * 0.9);
        float urough = max(0.001, roughness / aspect);
        float vrough = max(0.001, roughness * aspect);
        
        distribution.alpha_x = urough;
        distribution.alpha_y = vrough;
        distribution.sample_visible_area = true;
        
        fresnel_blend.Init(Rd, Rs, distribution, normal_);
    }
    
    uint Flags()
    {
        return BSDF_GlossyReflection;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return fresnel_blend.Eval(woW, wiW, mode);
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return fresnel_blend.PDF(woW, wiW, mode, flags);
    }
    
    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        return fresnel_blend.Samplef(woW, uc, u, mode, flags);
    }
};

#endif