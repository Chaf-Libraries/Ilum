#ifndef MICROFACET_REFLECTION_BSDF_HLSLI
#define MICROFACET_REFLECTION_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

template<typename Fresnel, typename Distribution>
struct MicrofacetReflectionBSDF
{
    float3 R;
    Fresnel fresnel;
    Distribution distribution;
    Frame frame;
    
    void Init(float3 R_, Distribution distribution_, Fresnel fresnel_, float3 normal_)
    {
        fresnel = fresnel_;
        distribution = distribution_;
        R = R_;
        frame.FromZ(normal_);
    }
    
    uint Flags()
    {
        return BSDF_GlossyReflection;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        
        float cos_theta_o = AbsCosTheta(wo);
        float cos_theta_i = AbsCosTheta(wi);

        float3 wm = wi + wo;

        if (cos_theta_i == 0.0 || cos_theta_o == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        if (IsBlack(wm))
        {
            return float3(0.0, 0.0, 0.0);
        }

        wm = normalize(wm);
        
        float D = distribution.D(wm);
        float G = distribution.G(wo, wi);
        float3 F = fresnel.Eval(dot(wi, Faceforward(wm, float3(0.0, 0.0, 1.0))));
        
        return R * D * G * F / (4.0 * cos_theta_i * cos_theta_o);
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!(flags & SampleFlags_Reflection) || !SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        
        float3 wm = normalize(wo + wi);
        
        return distribution.Pdf(wo, wm) / (4.0 * dot(wo, wm));
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
        
        if (wo.z < 0.0)
        {
            return bsdf_sample;
        }
        
        float3 wm = distribution.SampleWm(wo, u);
   
        if (dot(wo, wm) < 0.0)
        {
            return bsdf_sample;
        }

        float3 wi = reflect(-wo, wm);

        if (!SameHemisphere(wo, wi))
        {
            return bsdf_sample;
        }
        
        float3 wiW = frame.ToWorld(wi);
        
        bsdf_sample.f = Eval(woW, wiW, mode);
        bsdf_sample.wiW = wiW;
        bsdf_sample.pdf = distribution.Pdf(wo, wm) / (4.f * dot(wo, wm));
        bsdf_sample.flags = BSDF_GlossyReflection;
        bsdf_sample.eta = 1;
        
        return bsdf_sample;
    }
};

#endif