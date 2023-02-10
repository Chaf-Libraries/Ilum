#ifndef THIN_DIELECTRIC_BSDF_HLSLI
#define THIN_DIELECTRIC_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct ThinDielectricBSDF
{
    float3 R;
    float3 T;
    float eta;

    void Init(float3 R_, float3 T_, float eta_)
    {
        R = R_;
        T = T_;
        eta = eta_;
    }

    uint Flags()
    {
        return BSDF_Reflection | BSDF_Transmission | BSDF_Specular;
    }

    float3 Eval(float3 wo, float3 wi, TransportMode mode)
    {
        return 0.f;
    }

    float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags)
    {
        return 0.f;
    }

    BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample;
        bsdf_sample.Init();

        float F = FresnelDielectric(AbsCosTheta(wo), eta);

        if(F < 1)
        {
            F += Sqr(1 - F) * F / (1 - Sqr(F));
        }

        float pr = F;
        float pt = 1 - F;

        if(!(flags & BSDF_Reflection))
        {
            pr = 0.f;
        }
        if(!(flags & BSDF_Transmission))
        {
            pt = 0.f;
        }
        if(pr == 0 && pt == 0)
        {
            return bsdf_sample;
        }

        if(uc < pr / (pr + pt))
        {
            float3 wi = float3(-wo.x, -wo.y, wo.z);
            
            bsdf_sample.f = R * F / AbsCosTheta(wi);
            bsdf_sample.wi = wi;
            bsdf_sample.pdf = pr / (pr + pt);
            bsdf_sample.flags = BSDF_SpecularReflection;
            bsdf_sample.eta = 1;

            return bsdf_sample;
        }
        else
        {
            float3 wi = -wo;

            bsdf_sample.f = T * (1 - F) / AbsCosTheta(wi);
            bsdf_sample.wi = wi;
            bsdf_sample.pdf = pt / (pr + pt);
            bsdf_sample.flags = BSDF_SpecularTransmission;
            bsdf_sample.eta = 1;

            return bsdf_sample;
        }

        return bsdf_sample;
    }
};

#endif