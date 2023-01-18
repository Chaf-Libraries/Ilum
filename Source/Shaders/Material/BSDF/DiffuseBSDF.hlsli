#ifndef DIFFUSE_BSDF_HLSLI
#define DIFFUSE_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct DiffuseBSDF
{
    float3 reflectance;
    Frame frame;

    void Init(float3 reflectance_, float3 normal_)
    {
        reflectance = reflectance_;
        frame.FromZ(normal_);
    }
    
    uint Flags()
    {
        return BSDF_DiffuseReflection;
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        return reflectance * InvPI;
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!(flags & SampleFlags_Reflection) || !SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        return AbsCosTheta(wi) * InvPI;
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        
        BSDFSample bsdf_sample;
        
        bsdf_sample.Init();
        
        if (!(flags & SampleFlags_Reflection))
        {
            return bsdf_sample;
        }

        float3 wi = SampleCosineHemisphere(u);
        
        if (wo.z < 0)
        {
            wi.z *= -1;
        }

        float pdf = AbsCosTheta(wi) * InvPI;

        bsdf_sample.f = reflectance * InvPI;
        bsdf_sample.wiW = frame.ToWorld(wi);
        bsdf_sample.pdf = pdf;
        bsdf_sample.flags = BSDF_DiffuseReflection;
        bsdf_sample.eta = 1;

        return bsdf_sample;
    }
};

#endif