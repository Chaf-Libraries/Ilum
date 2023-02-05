#ifndef LAMBERTIAN_REFLECTION_BSDF_HLSLI
#define LAMBERTIAN_REFLECTION_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct LambertianReflectionBSDF
{
    float3 R;
    Frame frame;

    void Init(float3 R_, float3 normal_)
    {
        R = R_;
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
        
        return R * InvPI;
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
        
        if (wo.z < 0.f)
        {
            wi.z *= -1.f;
        }

        float3 wiW = frame.ToWorld(wi);
        
        bsdf_sample.f = Eval(woW, wiW, mode);
        bsdf_sample.wiW = wiW;
        bsdf_sample.pdf = PDF(woW, wiW, mode, flags);
        bsdf_sample.flags = BSDF_DiffuseReflection;
        bsdf_sample.eta = 1;

        return bsdf_sample;
    }
};

#endif