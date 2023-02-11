#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

#include "BSDF/BSDF.hlsli"
#include "../Interaction.hlsli"

struct BSDF
{
    void Init()
    {
        
    }
    
    uint Flags()
    {
        return 0;
    }

    float3 Eval(float3 wo, float3 wi, TransportMode mode)
    {
        return 0;
    }

    float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags)
    {
        return 0;
    }

    BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample sample_;
        return sample_;
    }
    
    // For GBuffer generation
    float GetRoughness()
    {
        return 0.f;
    }
    
    float3 GetEmissive()
    {
        return 0.f;
    }
};

struct Material
{
    BSDF bsdf;
    
    void Init(SurfaceInteraction surface_interaction)
    {
        bsdf.Init();
    }
};

#endif