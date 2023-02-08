#ifndef CONDUCTOR_BSDF_HLSLI
#define CONDUCTOR_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct ConductorBSDF
{
    TrowbridgeReitzDistribution distribution;
    float3 eta;
    float3 k;
    float3 R;
    Frame frame;
    
    void Init(float3 R_, float roughness, float3 eta_, float3 k_, float3 normal)
    {
        R = R_;
        eta = eta_;
        k = k_;
        distribution.alpha_x = roughness;
        distribution.alpha_y = roughness;
        distribution.sample_visible_area = true;
        frame.FromZ(normal);
    }

    uint Flags()
    {
        return distribution.EffectivelySmooth() ? BSDF_SpecularReflection : BSDF_GlossyReflection;
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);

        if(!SameHemisphere(wo, wi))
        {
            return 0.f;
        }

        if(distribution.EffectivelySmooth())
        {
            return 0.f;
        }

        float cos_theta_o = AbsCosTheta(wo);
        float cos_theta_i = AbsCosTheta(wi);

        if(cos_theta_i == 0 || cos_theta_o == 0)
        {
            return 0.f;
        }

        float3 wm = wi + wo;

        if(LengthSquared(wm) == 0)
        {
            return 0.f;
        }

        wm = normalize(wm);

        float3 F = FresnelConductor(abs(dot(wo, wm)), eta, k);

        return R * distribution.D(wm) * F * distribution.G(wo, wi) / (4 * cos_theta_i * cos_theta_o);
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);

        if(!(flags & SampleFlags_Reflection))
        {
            return 0.f;
        }
        if(!SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        if(distribution.EffectivelySmooth())
        {
            return 0.f;
        }

        float3 wm = wo + wi;

        if(LengthSquared(wm) == 0)
        {
            return 0.f;
        }

        wm = Faceforward(normalize(wm), float3(0, 0, 1));
        return distribution.Pdf(wo, wm) / (4 * abs(dot(wo, wm)));
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);

        BSDFSample bsdf_sample;
        bsdf_sample.Init();

        if(!(flags & SampleFlags_Reflection))
        {
            return bsdf_sample;
        }

        if(distribution.EffectivelySmooth())
        {
            float3 wi = float3(-wo.x, -wo.y, wo.z);
            
            bsdf_sample.f = R * FresnelConductor(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);
            bsdf_sample.wiW = frame.ToWorld(wi);
            bsdf_sample.pdf = 1;
            bsdf_sample.flags = BSDF_SpecularReflection;
            bsdf_sample.eta = 1;

            return bsdf_sample;
        }

        if(wo.z == 0)
        {
            return bsdf_sample;
        }

        float3 wm = distribution.SampleWm(wo, u);
        float3 wi = Reflect(wo, wm);

        if(!SameHemisphere(wo, wi))
        {
            return bsdf_sample;
        }

        float cos_theta_o = AbsCosTheta(wo);
        float cos_theta_i = AbsCosTheta(wi);
        if(cos_theta_i == 0 || cos_theta_o == 0)
        {
           return bsdf_sample;
        }

        float3 F = FresnelConductor(abs(dot(wo, wm)), eta, k);

        bsdf_sample.f = R * distribution.D(wm) * F * distribution.G(wo, wi) / (4 * cos_theta_i * cos_theta_o);
        bsdf_sample.wiW = frame.ToWorld(wi);
        bsdf_sample.pdf = distribution.Pdf(wo, wm) / (4 * abs(dot(wo, wm)));
        bsdf_sample.flags = BSDF_GlossyReflection;
        bsdf_sample.eta = 1;

        return bsdf_sample;
    }
};

#endif