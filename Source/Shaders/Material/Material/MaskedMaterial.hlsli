#ifndef MASK_MATERIAL_HLSLI
#define MASK_MATERIAL_HLSLI

#include "../BSDF/BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

template<typename T>
struct MaskedMaterial
{
    T material;
    float alpha;
    float threshold;

    float3 GetEmissive()
    {
        return material.GetEmissive();
    }

    void Init(T material_, float alpha_, float threshold_)
    {
        material = material_;
        alpha = alpha_;
        threshold = threshold_;
    }

    uint Flags()
    {
        return threshold < alpha ? material.Flags() : BSDF_Transmission;
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return threshold < alpha ?
            material.Eval(woW, wiW, mode) :
            0.f;
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return threshold < alpha ?
            material.PDF(woW, wiW, mode, flags) :
            0.f;
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample;
        if(threshold < alpha)
        {
            bsdf_sample = material.Samplef(woW, uc, u, mode, flags);
        }
        else
        {
            float3 wi = material.frame.ToLocal(-woW);
            bsdf_sample.f = 1.f / abs(CosTheta(wi));
            bsdf_sample.wiW = -woW;
            bsdf_sample.pdf = 1.f;
            bsdf_sample.flags = BSDF_SpecularTransmission;
            bsdf_sample.eta = 1.f;
        }
        return bsdf_sample;
    }
};

#endif