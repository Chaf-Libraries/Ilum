#ifndef GLASS_BSDF_HLSLI
#define GLASS_BSDF_HLSLI

#include "BSDF.hlsli"
#include "MicrofacetReflectionBSDF.hlsli"
#include "MicrofacetTransmissionBSDF.hlsli"
#include "FresnelSpecularBSDF.hlsli"
#include "../Fresnel.hlsli"
#include "../Scattering.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct GlassBSDF
{
    FresnelSpecularBSDF fresnel_specular;
    MicrofacetTransmissionBSDF<FresnelDielectric, TrowbridgeReitzDistribution> microfacet_transmission;
    MicrofacetReflectionBSDF <FresnelDielectric, TrowbridgeReitzDistribution>microfacet_reflection;
    bool is_specular;
    bool has_reflection;
    bool has_transmission;
    Frame frame;
    TrowbridgeReitzDistribution distribution;
    FresnelDielectric fresnel;
    
    void Init(float3 R, float3 T, float roughness, float refraction, float anisotropic, float3 normal)
    {
        float aspect = sqrt(1.0 - anisotropic * 0.9);
        float urough = roughness / aspect;
        float vrough = roughness * aspect;
        
        frame.FromZ(normal);
        
        is_specular = ((urough == 0 && vrough == 0) || (refraction == 1));
        
        if (is_specular)
        {
            fresnel_specular.Init(R, T, 1.f, refraction, normal);
        }
        else
        {
            has_reflection = !IsBlack(R);
            has_transmission = !IsBlack(T);
            
            distribution.alpha_x = urough;
            distribution.alpha_y = vrough;
            distribution.sample_visible_area = true;
            
            fresnel.eta_i = 1.f;
            fresnel.eta_t = refraction;

            if (has_reflection)
            {
                microfacet_reflection.Init(R, distribution, fresnel, normal);
            }
            
            if (has_transmission)
            {
                microfacet_transmission.Init(T, 1.f, refraction, distribution, fresnel, normal);
            }
        }
    }
    
    uint Flags()
    {
        uint flags;
        if (is_specular)
        {
            flags = BSDF_SpecularReflection | BSDF_SpecularTransmission;
        }
        else
        {
            if (has_reflection)
            {
                flags |= BSDF_GlossyReflection;
            }
            
            if (has_transmission)
            {
                flags |= BSDF_GlossyTransmission;
            }
        }
        return flags;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (is_specular)
        {
            return fresnel_specular.Eval(woW, wiW, mode);
        }
        else
        {
            if (has_reflection && SameHemisphere(wi, wo))
            {
                return microfacet_reflection.Eval(woW, wiW, mode);
            }
            else if (has_transmission && !SameHemisphere(wi, wo))
            {
                return microfacet_transmission.Eval(woW, wiW, mode);
            }
        }
        
        return 0.f;
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (is_specular)
        {
            return fresnel_specular.PDF(woW, wiW, mode, flags);
        }
        else
        {
            if (has_reflection && SameHemisphere(wi, wo))
            {
                return microfacet_reflection.PDF(woW, wiW, mode, flags);
            }
            else if (has_transmission && !SameHemisphere(wi, wo))
            {
                return microfacet_transmission.PDF(woW, wiW, mode, flags);
            }
        }
        
        return 0.f;
    }
    
    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        
        BSDFSample bsdf_sample;
        
        bsdf_sample.Init();
        
        if (is_specular)
        {
            bsdf_sample = fresnel_specular.Samplef(woW, uc, u, mode, flags);
        }
        else
        {
            if (has_reflection && has_transmission)
            {
                if ((uc + u.x + u.y) * 0.33f < 0.5f)
                {
                    bsdf_sample = microfacet_reflection.Samplef(woW, uc, u, mode, flags);
                    bsdf_sample.pdf *= 0.5f;
                    return bsdf_sample;

                }
                else
                {
                    bsdf_sample = microfacet_transmission.Samplef(woW, uc, u, mode, flags);
                    bsdf_sample.pdf *= 0.5f;
                    return bsdf_sample;
                }
            }
            else if (has_reflection)
            {
                return microfacet_reflection.Samplef(woW, uc, u, mode, flags);
            }
            else if (has_transmission)
            {
                return microfacet_transmission.Samplef(woW, uc, u, mode, flags);
            }
        }
        
        return bsdf_sample;
    }
};

#endif