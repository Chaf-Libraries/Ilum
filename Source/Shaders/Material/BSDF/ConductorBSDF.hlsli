#ifndef CONDUCTOR_BSDF_HLSLI
#define CONDUCTOR_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct ConductorBSDF
{
    Frame frame;
    TrowbridgeReitzDistribution distrib;
    float eta;
    float k;

    void Init(float roughness_u, float roughness_v, float eta_, float k_, float3 normal_)
    {
        distrib.Init(distrib.RoughnessToAlpha(roughness_u), distrib.RoughnessToAlpha(roughness_v));
        eta = eta_;
        k = k_;
        frame.FromZ(normal_);
    }
    
    uint Flags()
    {
        return distrib.EffectivelySmooth() ? BSDF_SpecularReflection
                                             : BSDF_GlossyReflection;
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!SameHemisphere(wo, wi))
        {
            return 0;
        }
        if (distrib.EffectivelySmooth())
        {
            return 0;
        }
        // Evaluate rough conductor BRDF
        // Compute cosines and $\wm$ for conductor BRDF
        float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        if (cosTheta_i == 0 || cosTheta_o == 0)
        {
            return 0;
        }
        
        float3 wm = wi + wo;
        if (LengthSquared(wm) == 0)
        {
            return 0;
        }
        wm = normalize(wm);

        // Evaluate Fresnel factor _F_ for conductor BRDF
        float F = FresnelComplex(abs(dot(wo, wm)), eta, k);

        return distrib.D(wm) * F * distrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!(flags & BSDF_Reflection))
        {
            return 0;
        }
        if (!SameHemisphere(wo, wi))
        {
            return 0;
        }
        if (distrib.EffectivelySmooth())
        {
            return 0;
        }
        // Evaluate sampling PDF of rough conductor BRDF
        float3 wm = wo + wi;
        if (LengthSquared(wm) == 0)
        {
            return 0;
        }
        wm = FaceForward(normalize(wm), normalize(float3(0, 0, 1)));
        return distrib.PDF(wo, wm) / (4 * abs(dot(wo, wm)));
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        
        BSDFSample bsdf_sample;
        
        bsdf_sample.Init();
        
        if (!(flags & BSDF_Reflection))
        {
            return bsdf_sample;
        }
        if (distrib.EffectivelySmooth())
        {
            // Sample perfect specular conductor BRDF
            float3 wi = float3(-wo.x, -wo.y, wo.z);
            
            bsdf_sample.f = FresnelComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);
            bsdf_sample.wiW = frame.ToWorld(wi);
            bsdf_sample.pdf = 1;
            bsdf_sample.flags = BSDF_SpecularReflection;
            
            return bsdf_sample;
        }
        // Sample rough conductor BRDF
        // Sample microfacet normal $\wm$ and reflected direction $\wi$
        if (wo.z == 0)
        {
            return bsdf_sample;
        }
        
        float3 wm = distrib.Sample_wm(wo, u);
        float3 wi = Reflect(wo, wm);
        if (!SameHemisphere(wo, wi))
        {
            return bsdf_sample;
        }
        // Compute PDF of _wi_ for microfacet reflection
        float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        if (cosTheta_i == 0 || cosTheta_o == 0)
        {
            return bsdf_sample;
        }
        // Evaluate Fresnel factor _F_ for conductor BRDF
        float F = FresnelComplex(abs(dot(wo, wm)), eta, k);

        bsdf_sample.f = distrib.D(wm) * F * distrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
        bsdf_sample.wiW = frame.ToWorld(wi);
        bsdf_sample.pdf = distrib.PDF(wo, wm) / (4 * abs(dot(wo, wm)));
        bsdf_sample.flags = BSDF_GlossyReflection;
    
        return bsdf_sample;
    }
};

#endif