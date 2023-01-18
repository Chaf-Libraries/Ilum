#ifndef DIFFUSE_TRANSMISSION_BSDF_HLSLI
#define DIFFUSE_TRANSMISSION_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct DiffuseTransmissionBSDF
{
    float3 reflectance;
    float3 transmission;
    Frame frame;

    void Init(float3 reflectance_, float3 transmission_, float3 normal_)
    {
        reflectance = reflectance_;
        transmission = transmission_;
        frame.FromZ(normal_);
    }
    
    uint Flags()
    {
        return BSDF_DiffuseTransmission;
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        return SameHemisphere(wo, wi) ? (reflectance * InvPI) : (transmission * InvPI);
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        float pr = max(reflectance.r, max(reflectance.g, reflectance.b));
        float pt = max(transmission.r, max(transmission.g, transmission.b));
        
        if (!(flags & BSDF_Reflection))
        {
            pr = 0;
        }
        
        if (!(flags & BSDF_Transmission))
        {
            pt = 0;
        }
        
        if (pr == 0 && pt == 0)
        {
            return 0.f;
        }

        if (SameHemisphere(wo, wi))
            return pr / (pr + pt) * AbsCosTheta(wi) * InvPI;
        else
            return pt / (pr + pt) * AbsCosTheta(wi) * InvPI;
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        
        BSDFSample bsdf_sample;
        bsdf_sample.Init();
        
        float pr = max(reflectance.r, max(reflectance.g, reflectance.b));
        float pt = max(transmission.r, max(transmission.g, transmission.b));
        
        if (!(flags & BSDF_Reflection))
        {
            pr = 0;
        }
        
        if (!(flags & BSDF_Transmission))
        {
            pt = 0;
        }
        
        if (pr == 0 && pt == 0)
        {
            return bsdf_sample;
        }
        
        if (uc < pr / (pr + pt))
        {
            // Sample diffuse BSDF reflection
            float3 wi = SampleCosineHemisphere(u);
            if (wo.z < 0)
            {
                wi.z *= -1;
            }
            bsdf_sample.wiW = frame.ToWorld(wi);
            bsdf_sample.f = Eval(woW, bsdf_sample.wiW, mode);
            bsdf_sample.pdf = AbsCosTheta(wi) * InvPI * pr / (pr + pt);
            bsdf_sample.flags = BSDF_DiffuseReflection;
            bsdf_sample.eta = 1;
        }
        else
        {
            // Sample diffuse BSDF transmission
            float3 wi = SampleCosineHemisphere(u);
            if (wo.z > 0)
            {
                wi.z *= -1;
            }
            bsdf_sample.wiW = frame.ToWorld(wi);
            bsdf_sample.f = Eval(woW, bsdf_sample.wiW, mode);
            bsdf_sample.pdf = AbsCosTheta(wi) * InvPI * pt / (pr + pt);
            bsdf_sample.flags = BSDF_DiffuseTransmission;
            bsdf_sample.eta = 1;
        }
        
        return bsdf_sample;
    }
};

#endif