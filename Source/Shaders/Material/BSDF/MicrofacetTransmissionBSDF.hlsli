#ifndef MICROFACET_TRANSMISSION_BSDF_HLSLI
#define MICROFACET_TRANSMISSION_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

//#define Distribution TrowbridgeReitzDistribution
//#define Fresnel FresnelDielectric
template<typename Fresnel, typename Distribution>
struct MicrofacetTransmissionBSDF
{
    float3 T;
    float etaA, etaB;
    Frame frame;
    Distribution distribution;
    Fresnel fresnel;
    
    void Init(float3 T_, float etaA_, float etaB_, Distribution distribution_, Fresnel fresnel_, float3 normal_)
    {
        T = T_;
        etaA = etaA_;
        etaB = etaB_;
        distribution = distribution_;
        fresnel = fresnel_;
        frame.FromZ(normal_);
    }
    
    uint Flags()
    {
        return BSDF_GlossyTransmission;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (SameHemisphere(wo, wi))
        {
            return 0.f;
        }

        float cos_theta_o = CosTheta(wo);
        float cos_theta_i = CosTheta(wi);

        if (cos_theta_o == 0.0 || cos_theta_i == 0.0)
        {
            return 0.f;
        }

        float eta = CosTheta(wo) > 0.0 ? (etaB / etaA) : (etaA / etaB);
        float3 wm = normalize(wo + wi * eta);

        if (wm.z < 0.0)
        {
            wm = -wm;
        }
        if (dot(wo, wm) * dot(wi, wm) > 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        
        float3 F = fresnel.Eval(dot(wo, wm));
        float D = distribution.D(wm);
        float G = distribution.G(wo, wi);
        
        float sqrt_denom = dot(wo, wm) + eta * dot(wi, wm);
        float factor = (mode == TransportMode_Radiance) ? (1.0 / eta) : 1.0;

        return (1.f - F) * T * abs(D * G * eta * eta * abs(dot(wi, wm)) * abs(dot(wo, wm)) * factor * factor / (cos_theta_i * cos_theta_o * sqrt_denom * sqrt_denom));
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!(flags & SampleFlags_Transmission) || SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        
        float eta = CosTheta(wo) > 0.0 ? (etaB / etaA) : (etaA / etaB);
        float3 wm = normalize(wo + wi * eta);

        if (dot(wo, wm) * dot(wi, wm) > 0.0)
        {
            return 0.0;
        }

        float sqrt_denom = dot(wo, wm) + eta * dot(wi, wm);
        float dwh_dwi = abs((eta * eta * dot(wi, wm)) / (sqrt_denom * sqrt_denom));
        float pdf = distribution.Pdf(wo, wm);

        return pdf * dwh_dwi;
    }
    
    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        
        BSDFSample bsdf_sample;
        
        bsdf_sample.Init();
        
        if (!(flags & SampleFlags_Transmission))
        {
            return bsdf_sample;
        }
        
        if (wo.z == 0.0)
        {
            return bsdf_sample;
        }
        
        bool entering = CosTheta(wo) > 0.0;
        float eta_i = entering ? etaA : etaB;
        float eta_t = entering ? etaB : etaA;

        float3 wm = distribution.SampleWm(wo, u);

        float3 wi = 0.f;
        if (!Refract(wo, Faceforward(wm, wo), eta_i / eta_t, wi))
        {
            return bsdf_sample;
        }

        float3 wiW = frame.ToWorld(wi);
        
        bsdf_sample.f = Eval(woW, wiW, mode);
        bsdf_sample.wiW = wiW;
        bsdf_sample.pdf = PDF(woW, wiW, mode, flags);
        bsdf_sample.flags = BSDF_GlossyTransmission;
        bsdf_sample.eta = eta_i / eta_t;
        
        return bsdf_sample;
    }
};

#endif