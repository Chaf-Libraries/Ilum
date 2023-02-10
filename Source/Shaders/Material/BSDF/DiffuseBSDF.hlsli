#ifndef DIFFUSE_BSDF_HLSLI
#define DIFFUSE_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct DiffuseBSDF
{
    float3 R;

    void Init(float3 R_)
    {
        R = R_;
    }
    
    uint Flags()
    {
        return BSDF_DiffuseReflection;
    }

    float3 Eval(float3 wo, float3 wi, TransportMode mode)
    {
        if (!SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        
        return R * InvPI;
    }

    float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags)
    {
        if (!(flags & SampleFlags_Reflection) || !SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        
        return AbsCosTheta(wi) * InvPI;
    }

    BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample;
        
        bsdf_sample.Init();
        
        if (!(flags & SampleFlags_Reflection))
        {
            return bsdf_sample;
        }

        float3 wi = SampleCosineHemisphere(u);
        
        if (wo.z < 0.f)
        {
            wi.z *= -1.f;
        }

        bsdf_sample.f = Eval(wo, wi, mode);
        bsdf_sample.wi = wi;
        bsdf_sample.pdf = PDF(wo, wi, mode, flags);
        bsdf_sample.flags = BSDF_DiffuseReflection;
        bsdf_sample.eta = 1;

        return bsdf_sample;
    }
};

#endif